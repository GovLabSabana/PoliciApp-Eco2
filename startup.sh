#!/bin/bash
echo "Verificando archivo FAISS..."
if [ ! -f "faiss_index/index.faiss" ] || [ ! -s "faiss_index/index.faiss" ]; then
    echo "Archivo FAISS no encontrado o vacío, descargando..."
    # Aquí puedes añadir código para descargar desde una fuente externa si es necesario
    
    # O reconstruir desde los fragmentos si están disponibles
    if [ -f "faiss_index/index.faiss.manifest" ]; then
        echo "Reconstruyendo desde fragmentos..."
        python split_faiss.py merge faiss_index/index.faiss.manifest
    fi
fi

# Iniciar la aplicación Streamlit
streamlit run app.py