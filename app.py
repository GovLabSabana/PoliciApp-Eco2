import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from html_template import logo
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import re
import speech_recognition as sr
import queue

# Load environment variables
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    st.error("Error: OPENAI_API_KEY not found in environment variables")
    st.stop()
    
st.set_page_config(page_title="EcoPoliciApp", layout="centered")



class LawDocumentProcessor:
    def __init__(self, document_directory="data", index_directory="faiss_index"):
        self.document_directory = document_directory
        self.index_directory = index_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        os.makedirs(self.document_directory, exist_ok=True)
        os.makedirs(self.index_directory, exist_ok=True)

    def load_vector_store(self):
        """Carga el vector store existente o lo descarga si no existe"""
        index_path = os.path.join(self.index_directory, "index.faiss")
    
        try:
            # Verificar si el archivo existe
            if not os.path.exists(index_path) or os.path.getsize(index_path) == 0:
                st.info("Descargando índice FAISS...")
            
                # Ejemplo usando requests para descargar desde URL pública
                import requests
                url = "URL_DE_TU_ARCHIVO_FAISS"
                response = requests.get(url)
            
                # Guardar el archivo descargado
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                with open(index_path, 'wb') as f:
                    f.write(response.content)
            
                st.success("Índice FAISS descargado correctamente.")
        
            # Cargar el índice
            if os.path.exists(index_path):
                return FAISS.load_local(
                    self.index_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            return None
        except Exception as e:
            st.error(f"Error cargando vector store: {str(e)}")
            return None
            
    def process_excel_file(self, file_path):
        """Procesa archivos Excel y los convierte en documentos para el vectorstore"""
        try:
            # Lee el archivo Excel (múltiples hojas)
            xls = pd.ExcelFile(file_path)
            documents = []
            
            # Procesa cada hoja como un documento separado
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Convertir el DataFrame a texto
                text_content = f"Hoja: {sheet_name}\n\n"
                
                # Agrega nombres de columnas
                text_content += " | ".join(df.columns) + "\n"
                
                # Agrega filas
                for _, row in df.iterrows():
                    text_content += " | ".join(str(cell) for cell in row) + "\n"
                
                # Crea un documento con el contenido de la hoja
                from langchain.schema import Document
                doc = Document(
                    page_content=text_content,
                    metadata={"source": os.path.basename(file_path), "sheet": sheet_name}
                )
                documents.append(doc)
                
            return documents
        except Exception as e:
            st.error(f"Error procesando archivo Excel {os.path.basename(file_path)}: {str(e)}")
            return []

    def process_documents(self):
        """Procesa los documentos PDF, Excel y TXT y crea el vector store"""
        try:
            # Busca todos los tipos de archivos soportados
            pdf_files = [f for f in os.listdir(self.document_directory) if f.lower().endswith('.pdf')]
            excel_files = [f for f in os.listdir(self.document_directory) if f.lower().endswith(('.xlsx', '.xls'))]
            txt_files = [f for f in os.listdir(self.document_directory) if f.lower().endswith('.txt')]
            
            all_files = pdf_files + excel_files + txt_files
            
            if not all_files:
                st.warning("No se encontraron archivos (PDF, Excel, TXT) en el directorio data.")
                return None

            documents = []
            successful_files = []
            failed_files = []
            
            # Procesar archivos PDF
            for pdf_file in pdf_files:
                try:
                    file_path = os.path.join(self.document_directory, pdf_file)
                    
                    # Verificar si el archivo es válido
                    with open(file_path, 'rb') as file:
                        header = file.read(5)
                        if header != b'%PDF-':
                            failed_files.append((pdf_file, "Encabezado PDF inválido"))
                            continue
                    
                    loader = PyPDFLoader(file_path)
                    doc_pages = loader.load()
                    
                    if doc_pages:
                        documents.extend(doc_pages)
                        successful_files.append(pdf_file)
                        st.success(f"✅ Procesado exitosamente: {pdf_file} ({len(doc_pages)} páginas)")
                    else:
                        failed_files.append((pdf_file, "No se pudo extraer contenido"))
                
                except Exception as e:
                    failed_files.append((pdf_file, str(e)))
                    continue
            
            # Procesar archivos Excel
            for excel_file in excel_files:
                try:
                    file_path = os.path.join(self.document_directory, excel_file)
                    excel_docs = self.process_excel_file(file_path)
                    
                    if excel_docs:
                        documents.extend(excel_docs)
                        successful_files.append(excel_file)
                        st.success(f"✅ Procesado exitosamente: {excel_file} ({len(excel_docs)} hojas)")
                    else:
                        failed_files.append((excel_file, "No se pudo extraer contenido"))
                
                except Exception as e:
                    failed_files.append((excel_file, str(e)))
                    continue
            
            # Procesar archivos TXT
            for txt_file in txt_files:
                try:
                    file_path = os.path.join(self.document_directory, txt_file)
                    loader = TextLoader(file_path)
                    txt_docs = loader.load()
                    
                    if txt_docs:
                        documents.extend(txt_docs)
                        successful_files.append(txt_file)
                        st.success(f"✅ Procesado exitosamente: {txt_file}")
                    else:
                        failed_files.append((txt_file, "No se pudo extraer contenido"))
                
                except Exception as e:
                    failed_files.append((txt_file, str(e)))
                    continue

            # Mostrar resumen de procesamiento
            st.write("---")
            st.write("📊 Resumen de procesamiento:")
            st.write(f"- Total archivos: {len(all_files)}")
            st.write(f"- Procesados correctamente: {len(successful_files)}")
            st.write(f"- Fallidos: {len(failed_files)}")
            
            if failed_files:
                st.error("❌ Archivos que no se pudieron procesar:")
                for file, error in failed_files:
                    st.write(f"- {file}: {error}")

            if not documents:
                st.warning("⚠️ No se pudo extraer contenido de ningún archivo.")
                return None

            texts = self.text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(texts, self.embeddings)
            vectorstore.save_local(self.index_directory)
            
            st.success(f"✅ Vector store creado exitosamente con {len(texts)} fragmentos de texto")
            return vectorstore
        
        except Exception as e:
            st.error(f"Error procesando documentos: {str(e)}")
            return None

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
        stream_handler = StreamHandler(response_placeholder)
        
        # Detectar tipo de consulta
        query_type = detect_query_type(prompt)
        
        # Obtener contexto legal
        legal_context = get_legal_context(vector_store, prompt)
        
        # Construir prompt enriquecido
        enhanced_prompt = f"""
        Tipo de consulta: {query_type}
        Consulta: {prompt}
        
        Contexto legal relevante:
        {format_legal_context(legal_context)}
        
        Proporciona una respuesta estructurada según el tipo de consulta:
        
        Para procedimientos operativos:
        - Pasos específicos de actuación
        - Base legal aplicable
        - Puntos críticos de control
        - Verificación en campo
        - Documentación requerida
        - Competencia policial
        - Coordinación necesaria
        
        Para consultas informativas:
        - Base legal aplicable
        - Puntos importantes
        - Referencias normativas
        """
        
        chat_model = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            api_key=API_KEY,
            streaming=True,
            callbacks=[stream_handler]
        )
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=enhanced_prompt)
        ]
        
        response = chat_model.invoke(messages)
        return stream_handler.text
            
    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")
        return "Lo siento, ocurrió un error al procesar su solicitud."

def detect_query_type(prompt):
    """Detecta el tipo de consulta para adaptar la respuesta"""
    keywords = {
        'PESCA': ['pesca', 'aunap', 'talla', 'peces'],
        'FLORA': ['madera', 'cubitaje', 'deforestación', 'quemas'],
        'FAUNA': ['animales', 'especies', 'tráfico', 'decomiso'],
        'MINERÍA': ['minería', 'maquinaria', 'extracción', 'explotación'],
        'RECURSOS_HÍDRICOS': ['agua', 'vertimientos', 'contaminación', 'río']
    }
    
    prompt_lower = prompt.lower()
    for category, terms in keywords.items():
        if any(term in prompt_lower for term in terms):
            return category
    return 'GENERAL'

def main():
    processor = LawDocumentProcessor()
    
    # Inicialización del vector store
    if os.path.exists(os.path.join("faiss_index", "index.faiss")):
        vector_store = processor.load_vector_store()
    else:
        st.warning("Procesando documentos legales por primera vez...")
        vector_store = processor.process_documents()
    
    if vector_store is None:
        st.error("No se pudo inicializar la base de conocimientos")
        st.stop()

    # UI Setup
    st.write(logo, unsafe_allow_html=True)
    st.title("EcoPoliciApp", anchor=False)
    st.markdown("**Asistente virtual para procedimientos ambientales y protección de recursos naturales**")
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.markdown("""
        **Sistema de Consulta Ambiental**
        
        Tipos de consultas:
        - Información sobre normativa ambiental
        - Procedimientos de protección
        - Infracciones y sanciones
        - Recomendaciones preventivas
        
        **Formatos de archivo soportados:**
        - PDF: Documentos legales, normativas, informes
        - Excel (.xlsx, .xls): Datos tabulares, registros
        - TXT: Documentos de texto plano
        """)
        
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("¿En qué puedo ayudarte?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👮"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = get_chat_response(prompt, vector_store)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()