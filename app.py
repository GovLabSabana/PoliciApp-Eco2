import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain.callbacks.base import BaseCallbackHandler
from html_template import logo
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import re
import time
import concurrent.futures
import numpy as np
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError

# Load environment variables
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    st.error("Error: OPENAI_API_KEY not found in environment variables")
    st.stop()
    
st.set_page_config(page_title="EcoPoliciApp", layout="centered")

# Constants for optimization
MAX_DOCUMENTS_FOR_EMBEDDING = 15000  # Limit maximum documents to process
CHUNK_SIZE = 800  # Reduced chunk size for better processing
CHUNK_OVERLAP = 150  # Reduced overlap
BATCH_SIZE = 50  # Reduced batch size for embedding creation
USE_LOCAL_EMBEDDINGS = True  # Set to False to use OpenAI embeddings

class LawDocumentProcessor:
    def __init__(self, document_directory="data", index_directory="faiss_index"):
        self.document_directory = document_directory
        self.index_directory = index_directory
        self.embeddings = self.create_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        os.makedirs(self.document_directory, exist_ok=True)
        os.makedirs(self.index_directory, exist_ok=True)
    
    def create_embeddings(self):
        """Create embeddings model with fallback options"""
        if USE_LOCAL_EMBEDDINGS:
            try:
                # Using a smaller, efficient local model for embeddings
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
            except Exception as e:
                st.warning(f"Error inicializando embeddings locales: {str(e)}. Usando OpenAI embeddings como respaldo.")
        
        # Fallback to OpenAI embeddings
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            retry=retry_if_exception_type(RateLimitError)
        )
        def _create_openai_embeddings():
            return OpenAIEmbeddings()
        
        try:
            return _create_openai_embeddings()
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return None

    def load_vector_store(self):
        """Loads existing vector store or creates a new one if it doesn't exist"""
        index_faiss_path = os.path.join(self.index_directory, "index.faiss")
        
        # Check if the FAISS file exists and has content
        faiss_exists = os.path.exists(index_faiss_path) and os.path.getsize(index_faiss_path) > 0
        
        if not faiss_exists:
            st.info("FAISS index no encontrado, creando nuevo índice...")
            return self.process_documents()
        
        # Try loading the vector store
        try:
            vector_store = FAISS.load_local(
                self.index_directory, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            st.error(f"Error cargando vector store: {str(e)}")
            
            # If loading fails, try rebuilding
            st.warning("El índice FAISS parece estar corrupto. Intentando reconstruirlo...")
            if os.path.exists(index_faiss_path):
                os.remove(index_faiss_path)
            
            # Create a new index from documents
            return self.process_documents()

    def process_large_csv_file(self, file_path, delimiter='|', sample_ratio=0.2):
        """Procesa archivos CSV con delimitador personalizado de manera más eficiente"""
        try:
            # Check file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
        
            st.info(f"Procesando archivo CSV: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")

            # For extremely large files, sample a percentage
            should_sample = file_size_mb > 100
            
            # Try to use pandas for more efficient CSV processing
            try:
                # For large files, read in chunks
                if should_sample:
                    # Estimate number of lines and calculate appropriate chunksize
                    st.info(f"Archivo muy grande: se procesará una muestra del {sample_ratio*100:.0f}%")
                    df = pd.read_csv(
                        file_path, 
                        sep=delimiter, 
                        encoding='utf-8', 
                        error_bad_lines=False,
                        warn_bad_lines=False,
                        low_memory=True,
                        memory_map=True,
                        sample=sample_ratio  # This is a custom parameter; pandas doesn't have this
                    )
                else:
                    df = pd.read_csv(
                        file_path, 
                        sep=delimiter, 
                        encoding='utf-8', 
                        error_bad_lines=False,
                        warn_bad_lines=False,
                        low_memory=True
                    )
                
                # Create documents from dataframe
                docs = []
                progress_bar = st.progress(0)
                
                # Convert each row to a document
                for i, row in enumerate(df.itertuples()):
                    # Update progress every 100 rows
                    if i % 100 == 0:
                        progress_bar.progress(min(i / len(df), 1.0))
                    
                    # Combine row values into text
                    row_text = " ".join([str(val) for val in row if pd.notna(val) and str(val).strip()])
                    
                    if len(row_text) > 10:  # Skip very short content
                        docs.append(
                            Document(
                                page_content=row_text,
                                metadata={
                                    "source": file_path,
                                    "row": i
                                }
                            )
                        )
                
                progress_bar.empty()
                return docs
                
            except Exception as pd_error:
                # Fallback to manual CSV processing
                st.warning(f"Error al procesar con pandas: {str(pd_error)}. Usando método alternativo.")
            
            # Process in batches to avoid memory issues
            docs = []
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Count approximate total lines for large files
            estimated_total_lines = 0
            if should_sample:
                # Estimate based on first 1MB
                sample_size = min(file_size_bytes, 1024 * 1024)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    sample = f.read(sample_size)
                    lines_in_sample = sample.count('\n')
                    estimated_total_lines = int((file_size_bytes / sample_size) * lines_in_sample)
                st.info(f"Archivo muy grande: se procesará una muestra del {sample_ratio*100:.0f}% (aprox. {int(estimated_total_lines*sample_ratio)} líneas)")
            else:
                # Count lines for smaller files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, _ in enumerate(f):
                        if i % 100000 == 0:  # Update progress periodically
                            progress_bar.progress(min(0.5 * i / 1000000, 0.5))  # Cap at 50%
                        estimated_total_lines = i + 1
            
            # Process file with sampling for large files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines_processed = 0
                current_doc_texts = []
                current_text = ""
                
                for line_num, line in enumerate(f):
                    # Apply sampling for large files
                    if should_sample and np.random.random() > sample_ratio:
                        continue
                        
                    lines_processed += 1
                    
                    # Update progress every 1000 processed lines
                    if lines_processed % 1000 == 0:
                        progress = min(0.5 + 0.5 * (lines_processed / (estimated_total_lines * (sample_ratio if should_sample else 1))), 1.0)
                        progress_bar.progress(progress)
                    
                    # Extract and process text from line
                    try:
                        # Process delimited fields more efficiently
                        fields = [field.strip() for field in line.strip().split(delimiter) if field.strip()]
                        
                        if fields:
                            # Consolidate text only if it has meaningful content
                            line_text = " ".join(fields)
                            if len(line_text) > 10:  # Skip very short lines
                                current_text += line_text + "\n"
                            
                            # Check if we've accumulated enough text for a document
                            if len(current_text) >= CHUNK_SIZE:
                                current_doc_texts.append(current_text)
                                current_text = ""
                                
                                # Process in batches to avoid memory issues
                                if len(current_doc_texts) >= 100:
                                    # Create document objects
                                    batch_docs = [
                                        Document(
                                            page_content=text,
                                            metadata={
                                                "source": file_path,
                                                "chunk": i
                                            }
                                        ) for i, text in enumerate(current_doc_texts)
                                    ]
                                    docs.extend(batch_docs)
                                    current_doc_texts = []
                                    
                    except Exception as line_error:
                        pass  # Skip problematic lines silently
                
                # Process any remaining text
                if current_text and len(current_text) > 50:  # Only keep if substantial
                    current_doc_texts.append(current_text)
                
                # Create documents from any remaining text chunks
                if current_doc_texts:
                    batch_docs = [
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "chunk": i + len(docs)
                            }
                        ) for i, text in enumerate(current_doc_texts)
                    ]
                    docs.extend(batch_docs)
                
            progress_bar.empty()
            
            # Apply document limit if needed
            if len(docs) > MAX_DOCUMENTS_FOR_EMBEDDING:
                st.warning(f"Limitando a {MAX_DOCUMENTS_FOR_EMBEDDING} documentos de {len(docs)} para optimizar rendimiento")
                # Keep a random sample, weighted toward the beginning of the file
                indices = np.random.choice(
                    range(len(docs)), 
                    size=MAX_DOCUMENTS_FOR_EMBEDDING, 
                    replace=False,
                    p=np.linspace(2, 1, len(docs)) / np.sum(np.linspace(2, 1, len(docs)))
                )
                docs = [docs[i] for i in sorted(indices)]
            
            st.success(f"Procesado completo: {len(docs)} documentos creados")
            return docs
    
        except Exception as e:
            st.error(f"Error procesando archivo CSV {os.path.basename(file_path)}: {str(e)}")
            return []

    def process_large_text_file(self, file_path):
        """Processes large text files in optimized chunks"""
        try:
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # For extremely large files, use sampling
            sample_ratio = 1.0
            if file_size_mb > 100:
                sample_ratio = 0.2
                st.info(f"Archivo muy grande ({file_size_mb:.2f} MB): procesando {sample_ratio*100:.0f}% del contenido")
            
            chunk_size = 100000  # 100KB chunks for reading
            docs = []
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                progress_bar = st.progress(0)
                
                bytes_read = 0
                chunks_text = ""
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    bytes_read += len(chunk)
                    progress_bar.progress(min(bytes_read / file_size, 1.0))
                    
                    # Apply sampling for large files
                    if file_size_mb > 100 and np.random.random() > sample_ratio:
                        continue
                        
                    chunks_text += chunk
                    
                    # Process accumulated text when it reaches target size
                    if len(chunks_text) >= CHUNK_SIZE * 5:
                        # Split into smaller chunks for better context preservation
                        text_chunks = self.text_splitter.split_text(chunks_text)
                        
                        # Create documents
                        for i, text_chunk in enumerate(text_chunks):
                            if len(text_chunk) >= 100:  # Only keep substantial chunks
                                docs.append(
                                    Document(
                                        page_content=text_chunk,
                                        metadata={
                                            "source": file_path,
                                            "chunk": len(docs) + i
                                        }
                                    )
                                )
                        
                        chunks_text = ""  # Reset accumulated text
            
            # Process any remaining text
            if chunks_text:
                text_chunks = self.text_splitter.split_text(chunks_text)
                for i, text_chunk in enumerate(text_chunks):
                    if len(text_chunk) >= 100:
                        docs.append(
                            Document(
                                page_content=text_chunk,
                                metadata={
                                    "source": file_path,
                                    "chunk": len(docs) + i
                                }
                            )
                        )
            
            progress_bar.empty()
            
            # Apply document limit if needed
            if len(docs) > MAX_DOCUMENTS_FOR_EMBEDDING:
                st.warning(f"Limitando a {MAX_DOCUMENTS_FOR_EMBEDDING} documentos de {len(docs)} para optimizar rendimiento")
                # Keep a stratified sample from beginning, middle and end
                indices = []
                # 40% from beginning
                indices.extend(list(range(0, int(0.4 * MAX_DOCUMENTS_FOR_EMBEDDING))))
                # 30% from middle
                mid_start = len(docs) // 2 - int(0.15 * MAX_DOCUMENTS_FOR_EMBEDDING)
                indices.extend(list(range(mid_start, mid_start + int(0.3 * MAX_DOCUMENTS_FOR_EMBEDDING))))
                # 30% from end
                end_start = max(0, len(docs) - int(0.3 * MAX_DOCUMENTS_FOR_EMBEDDING))
                indices.extend(list(range(end_start, len(docs))))
                # Ensure we don't exceed our limit
                indices = indices[:MAX_DOCUMENTS_FOR_EMBEDDING]
                docs = [docs[i] for i in sorted(indices)]
            
            return docs
        
        except Exception as e:
            st.error(f"Error procesando archivo de texto grande {os.path.basename(file_path)}: {str(e)}")
            return []

    def process_pdf_file(self, file_path):
        """Process PDF files with optimizations"""
        try:
            # Check if file is valid PDF
            with open(file_path, 'rb') as file:
                header = file.read(5)
                if header != b'%PDF-':
                    return [], f"Encabezado PDF inválido"
            
            # Get file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # For very large PDFs, use page sampling
            sample_pages = False
            sample_ratio = 1.0
            
            if file_size_mb > 50:
                sample_pages = True
                sample_ratio = 0.3
                st.info(f"PDF grande ({file_size_mb:.2f} MB): procesando aprox. {sample_ratio*100:.0f}% de las páginas")
            
            loader = PyPDFLoader(file_path)
            doc_pages = loader.load()
            
            # Apply sampling for large PDFs
            if sample_pages and len(doc_pages) > 20:
                # Keep first few pages (table of contents, etc)
                keep_pages = [doc_pages[i] for i in range(min(5, len(doc_pages)))]
                
                # Sample from the rest
                remaining = list(range(5, len(doc_pages)))
                num_to_sample = int(len(remaining) * sample_ratio)
                
                if num_to_sample > 0:
                    sampled = np.random.choice(remaining, size=num_to_sample, replace=False)
                    sampled_pages = [doc_pages[i] for i in sampled]
                    doc_pages = keep_pages + sampled_pages
            
            return doc_pages, None
            
        except Exception as e:
            return [], str(e)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(RateLimitError)
    )
    def process_documents(self):
        """Procesa los documentos PDF, Excel y TXT y crea el vector store de manera optimizada"""
        try:
            # Busca todos los tipos de archivos soportados
            pdf_files = [f for f in os.listdir(self.document_directory) if f.lower().endswith('.pdf')]
            txt_files = [f for f in os.listdir(self.document_directory) if f.lower().endswith('.txt')]
            
            all_files = pdf_files + txt_files
            
            if not all_files:
                st.warning("No se encontraron archivos en el directorio data.")
                return None

            documents = []
            successful_files = []
            failed_files = []
            
            # Mostrar progreso global
            progress_bar_all = st.progress(0)
            total_files = len(all_files)
            files_processed = 0
            
            # Procesar archivos en paralelo con ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Procesar archivos PDF
                pdf_futures = {}
                for pdf_file in pdf_files:
                    file_path = os.path.join(self.document_directory, pdf_file)
                    pdf_futures[executor.submit(self.process_pdf_file, file_path)] = pdf_file
                
                # Procesar resultados de PDFs
                for future in concurrent.futures.as_completed(pdf_futures):
                    pdf_file = pdf_futures[future]
                    file_path = os.path.join(self.document_directory, pdf_file)
                    
                    try:
                        doc_pages, error = future.result()
                        
                        if error is None and doc_pages:
                            documents.extend(doc_pages)
                            successful_files.append(pdf_file)
                            st.success(f"✅ Procesado exitosamente: {pdf_file} ({len(doc_pages)} páginas)")
                        else:
                            failed_files.append((pdf_file, error or "No se pudo extraer contenido"))
                    
                    except Exception as e:
                        failed_files.append((pdf_file, str(e)))
                    
                    # Update overall progress
                    files_processed += 1
                    progress_bar_all.progress(files_processed / total_files)
                
                # Procesar archivos TXT
                txt_futures = {}
                for txt_file in txt_files:
                    file_path = os.path.join(self.document_directory, txt_file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    
                    # Process differently based on file size and content
                    if file_size > 50:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            try:
                                first_line = f.readline().strip()
                                if '|' in first_line:
                                    # Process as CSV with pipe delimiter
                                    txt_futures[executor.submit(self.process_large_csv_file, file_path, '|')] = txt_file
                                else:
                                    # Process as large text file
                                    txt_futures[executor.submit(self.process_large_text_file, file_path)] = txt_file
                            except:
                                # Default to text processing if reading fails
                                txt_futures[executor.submit(self.process_large_text_file, file_path)] = txt_file
                    else:
                        # Regular text loader for smaller files
                        def load_text(path):
                            try:
                                loader = TextLoader(path, encoding='utf-8')
                                return loader.load()
                            except Exception as e:
                                return [], str(e)
                        
                        txt_futures[executor.submit(load_text, file_path)] = txt_file
                
                # Process text file results
                for future in concurrent.futures.as_completed(txt_futures):
                    txt_file = txt_futures[future]
                    
                    try:
                        txt_docs = future.result()
                        
                        if isinstance(txt_docs, tuple):
                            txt_docs, error = txt_docs
                            if error:
                                failed_files.append((txt_file, error))
                        
                        if txt_docs:
                            documents.extend(txt_docs)
                            successful_files.append(txt_file)
                            st.success(f"✅ Procesado exitosamente: {txt_file} ({len(txt_docs)} fragmentos)")
                        else:
                            failed_files.append((txt_file, "No se pudo extraer contenido"))
                    
                    except Exception as e:
                        failed_files.append((txt_file, str(e)))
                    
                    # Update overall progress
                    files_processed += 1
                    progress_bar_all.progress(files_processed / total_files)
            
            # Clear overall progress bar
            progress_bar_all.empty()

            # Mostrar resumen de procesamiento
            st.write("---")
            st.write("📊 Resumen de procesamiento:")
            st.write(f"- Total archivos: {len(all_files)}")
            st.write(f"- Procesados correctamente: {len(successful_files)}")
            st.write(f"- Fallidos: {len(failed_files)}")
            
            if failed_files:
                with st.expander("❌ Archivos que no se pudieron procesar"):
                    for file, error in failed_files:
                        st.write(f"- {file}: {error}")

            if not documents:
                st.warning("⚠️ No se pudo extraer contenido de ningún archivo.")
                return None

            # Apply document limit if needed to prevent memory issues
            total_docs = len(documents)
            if total_docs > MAX_DOCUMENTS_FOR_EMBEDDING:
                st.warning(f"Limitando a {MAX_DOCUMENTS_FOR_EMBEDDING} documentos de {total_docs} para optimizar rendimiento")
                # Stratified sampling to keep representative documents
                indices = []
                files_dict = {}
                
                # Group by source file
                for i, doc in enumerate(documents):
                    source = doc.metadata.get('source', 'unknown')
                    if source not in files_dict:
                        files_dict[source] = []
                    files_dict[source].append(i)
                
                # Sample from each file proportionally
                for source, indices_list in files_dict.items():
                    file_proportion = len(indices_list) / total_docs
                    n_to_sample = max(1, int(file_proportion * MAX_DOCUMENTS_FOR_EMBEDDING))
                    if len(indices_list) <= n_to_sample:
                        indices.extend(indices_list)
                    else:
                        sampled = np.random.choice(indices_list, size=n_to_sample, replace=False)
                        indices.extend(sampled)
                
                # Ensure we don't exceed our limit
                indices = indices[:MAX_DOCUMENTS_FOR_EMBEDDING]
                documents = [documents[i] for i in sorted(indices)]
            
            # Split documents into smaller chunks for better context preservation
            st.info(f"Dividiendo {len(documents)} documentos en fragmentos más pequeños...")
            all_texts = self.text_splitter.split_documents(documents)
            st.success(f"Obtenidos {len(all_texts)} fragmentos de texto")
            
            # Set up progress tracking for embeddings
            st.info("Generando embeddings para los documentos...")
            progress_bar = st.progress(0)
            total_texts = len(all_texts)
            
            # Process in smaller batches
            vectorstore = None
            
            # Limit documents if there are still too many after splitting
            if len(all_texts) > MAX_DOCUMENTS_FOR_EMBEDDING:
                st.warning(f"Limitando a {MAX_DOCUMENTS_FOR_EMBEDDING} fragmentos de {len(all_texts)}")
                # Random sample with bias toward the beginning of the collection
                weights = np.linspace(2, 1, len(all_texts))
                weights = weights / np.sum(weights)
                indices = np.random.choice(
                    range(len(all_texts)), 
                    size=min(MAX_DOCUMENTS_FOR_EMBEDDING, len(all_texts)), 
                    replace=False,
                    p=weights
                )
                all_texts = [all_texts[i] for i in sorted(indices)]
            
            # Create embeddings in batches
            for i in range(0, len(all_texts), BATCH_SIZE):
                batch = all_texts[i:i+BATCH_SIZE]
                
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, self.embeddings)
                else:
                    temp_vectorstore = FAISS.from_documents(batch, self.embeddings)
                    vectorstore.merge_from(temp_vectorstore)
                
                # Update progress
                progress = min((i + BATCH_SIZE) / total_texts, 1.0)
                progress_bar.progress(progress)
                
                # Small delay to prevent rate limits
                time.sleep(0.5)
            
            progress_bar.empty()
            
            # Save the vector store
            st.info("Guardando índice FAISS...")
            vectorstore.save_local(self.index_directory)
            
            st.success(f"✅ Vector store creado exitosamente con {len(all_texts)} fragmentos de texto")
            return vectorstore
        
        except Exception as e:
            st.error(f"Error procesando documentos: {str(e)}")
            return None

# Remaining code for setup_retrieval_chain, StreamHandler, get_legal_context, etc.
# remains the same as in your original code

def setup_retrieval_chain(vector_store):
    """Configura la cadena de recuperación para consultas"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return retrieval_chain

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def get_legal_context(vector_store, query):
    """Obtiene el contexto legal relevante para una consulta"""
    similar_docs = vector_store.similarity_search(query, k=5)
    context = []
    
    for doc in similar_docs:
        content = doc.page_content
        source = doc.metadata.get('source', 'Documento desconocido')
        page = doc.metadata.get('page', 'N/A')
        sheet = doc.metadata.get('sheet', None)
        
        # Extraer referencias legales específicas
        legal_refs = re.findall(r'(?:Artículo|Art\.|Ley|Decreto)\s+\d+[^\n]*', content)
        
        # Preparar información de la fuente
        source_info = f"{source}"
        if page != 'N/A':
            source_info += f" (Pág. {page})"
        if sheet:
            source_info += f" (Hoja: {sheet})"
        
        context.append({
            'source': source_info,
            'content': content,
            'legal_refs': legal_refs
        })
    
    return context

SYSTEM_PROMPT = """
Eres EcoPoliciApp, un asistente especializado en legislación ambiental colombiana, enfocado en apoyar a oficiales de la Policía Ambiental y de Carabineros.

ÁREAS DE ESPECIALIZACIÓN:

🐟 PESCA:
- Regulaciones AUNAP
- Tallas mínimas permitidas
- Vedas y restricciones

🌳 FLORA:
- Identificación de madera
- Cálculo de cubitaje
- Deforestación ilegal
- Quemas controladas

🦁 FAUNA:
- Tráfico de especies
- Manejo en desastres
- Protocolos de decomiso
- Especies protegidas

⛏️ MINERÍA:
- Licencias y permisos
- Procedimientos de control
- Maquinaria autorizada
- Protocolos de incautación

🌊 RECURSOS HÍDRICOS:
- Contaminación
- Vertimientos
- Protección de cuencas

FORMATO DE RESPUESTA:

📋 PROCEDIMIENTO OPERATIVO:
• [Acciones paso a paso]

⚖️ BASE LEGAL:
• [Referencias normativas específicas]

🚨 PUNTOS CRÍTICOS:
• [Aspectos clave a verificar]

🔍 VERIFICACIÓN EN CAMPO:
• [Lista de chequeo]

📄 DOCUMENTACIÓN REQUERIDA:
• [Documentos necesarios]

👮 COMPETENCIA POLICIAL:
• [Alcance de la autoridad]

🤝 COORDINACIÓN INSTITUCIONAL:
• [Entidades a contactar]

DIRECTRICES:
1. Priorizar seguridad del personal
2. Proteger evidencia
3. Documentar hallazgos
4. Coordinar con autoridades competentes
"""

def format_legal_context(context):
    """Formatea el contexto legal para el prompt"""
    formatted = []
    for item in context:
        refs = '\n'.join(f"• {ref}" for ref in item['legal_refs']) if item['legal_refs'] else "No se encontraron referencias específicas"
        formatted.append(f"""
        📚 Fuente: {item['source']}
        
        ⚖️ Referencias legales:
        {refs}
        
        💡 Contexto relevante:
        {item['content'][:500]}...
        """)
    return '\n'.join(formatted)

def get_chat_response(prompt, vector_store, temperature=0.3):
    """Genera respuesta considerando el contexto legal"""
    try:
        response_placeholder = st.empty()
        
        # Get relevant legal context
        legal_context = get_legal_context(vector_store, prompt)
        formatted_context = format_legal_context(legal_context)
        
        # Set up the chat handler
        stream_handler = StreamHandler(response_placeholder)
        
        # Create ChatOpenAI with streaming
        chat = ChatOpenAI(
            temperature=temperature,
            streaming=True,
            callbacks=[stream_handler]
        )
        
        # Create messages with system prompt, context and user query
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"""
            CONSULTA DEL USUARIO: {prompt}
            
            CONTEXTO LEGAL RELEVANTE:
            {formatted_context}
            
            Por favor, responde a la consulta usando el formato especificado en tus instrucciones.
            """)
        ]
        
        # Get response
        response = chat(messages)
        return response.content, legal_context
    
    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")
        return f"Lo siento, ocurrió un error: {str(e)}", []

def main():
    """Función principal de la aplicación"""
    # Display app header
    st.markdown(logo, unsafe_allow_html=True)
    st.title("🌿 EcoPoliciApp")
    st.subheader("Asistente en legislación ambiental para la Policía Nacional")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processor' not in st.session_state:
        st.session_state.processor = LawDocumentProcessor()
    
    if 'vector_store' not in st.session_state:
        with st.spinner("Cargando base de conocimiento..."):
            st.session_state.vector_store = st.session_state.processor.load_vector_store()
    
    if 'last_sources' not in st.session_state:
        st.session_state.last_sources = []
    
    # Check if vector store was loaded successfully
    if st.session_state.vector_store is None:
        st.error("Error: No se pudo cargar la base de conocimiento. Por favor, asegúrate de tener documentos en la carpeta 'data'.")
        st.stop()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Text input
    user_input = st.chat_input("Escribe tu consulta sobre legislación ambiental...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        with st.chat_message("assistant"):
            response_content, sources = get_chat_response(
                user_input, 
                st.session_state.vector_store
            )
        
        # Save sources for display
        st.session_state.last_sources = sources
        
        # Add to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_content
        })
        
        # Display sources
        if st.session_state.last_sources:
            with st.expander("📚 Fuentes consultadas"):
                for i, source in enumerate(st.session_state.last_sources):
                    st.markdown(f"**Fuente {i+1}:** {source['source']}")
                    if source['legal_refs']:
                        st.markdown("**Referencias legales:**")
                        for ref in source['legal_refs']:
                            st.markdown(f"- {ref}")
    
    # Sidebar
    with st.sidebar:
        st.header("Opciones")
        
        # Clear chat
        if st.button("🗑️ Limpiar chat"):
            st.session_state.chat_history = []
            st.session_state.last_sources = []
            st.rerun()
        
        # Document management
        st.header("Gestión de documentos")
        st.write("Para agregar documentos a la base de conocimiento, colócalos en la carpeta 'data'.")
        
        # Regenerate index
        if st.button("🔄 Regenerar índice de documentos"):
            with st.spinner("Regenerando índice de documentos..."):
                st.session_state.vector_store = st.session_state.processor.process_documents()
                st.success("¡Índice regenerado con éxito!")
        
        # About section
        st.header("Acerca de EcoPoliciApp")
        st.write("""
        EcoPoliciApp es un asistente especializado para la Policía Ambiental y de Carabineros, 
        que proporciona información precisa sobre normativas ambientales colombianas.
        
        Desarrollado para apoyar la labor policial en:
        - Protección de flora y fauna
        - Control de pesca ilegal
        - Minería
        - Recursos hídricos
        """)

if __name__ == "__main__":
    main()