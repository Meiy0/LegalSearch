import os
from flask import Flask, render_template, request, redirect, url_for, session, Response, make_response, jsonify
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import uuid
from cachetools import LRUCache
from sklearn.metrics.pairwise import cosine_similarity
import io
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
import json


#--- Inicialización ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'f3b0c3f0b0c3f0b0c3f0b0c3f0b0c3f0'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

#--- Carga del Modelo ---
try:
    print("Cargando modelo de SentenceTransformer...")
    modelo = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    modelo = None

#--- Caché Global ---
PDF_CACHE = LRUCache(maxsize=20) 

#--- Funciones de Lógica ---

#Extrae y divide el pdf en fragmentos
def extraer_y_dividir_por_pagina(stream_de_bytes):
    MIN_CHAR_LENGTH = 150
    datos_fragmentos = []
    
    with fitz.open(stream=stream_de_bytes, filetype="pdf") as doc:
        for num_pagina, pagina in enumerate(doc, start=1):
            texto_pagina = pagina.get_text()
            
            fragmentos_pagina = re.split(r'\n\s*\n', texto_pagina)
            
            for f in fragmentos_pagina:
                f_limpio = f.strip()
                if f_limpio and len(f_limpio) > MIN_CHAR_LENGTH:
                    datos_fragmentos.append({
                        "texto": f_limpio,
                        "pagina": num_pagina 
                    })
                    
    return datos_fragmentos

#Crea indice FAISS
def crear_indice_faiss(datos_fragmentos, modelo):
    if not datos_fragmentos or modelo is None:
        return None, []
    
    textos_para_embed = [item['texto'] for item in datos_fragmentos]
    
    embeddings = modelo.encode(textos_para_embed, convert_to_tensor=False)
    embeddings_np = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_np)
    
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    return index, datos_fragmentos

#Busca la consulta y devulve la respuesta junto con la página
def buscar_en_indice(consulta, index, datos_fragmentos, modelo, k=3):
    if index is None or modelo is None:
        return []
        
    vector_consulta = modelo.encode([consulta])
    vector_consulta_np = np.array(vector_consulta).astype('float32')
    faiss.normalize_L2(vector_consulta_np)
    
    distances, indices = index.search(vector_consulta_np, k)
    
    resultados = []
    for i in indices[0]:
        if i != -1:
            resultados.append(datos_fragmentos[i])
            
    return resultados


#--- Funciones de Lógica Clasificación Múltiple ---

stemmer = nltk.SnowballStemmer("spanish")
stop_words_es = set(stopwords.words("spanish"))

def preprocesar_texto(texto):
    texto_minusculas = texto.lower()
    tokens = word_tokenize(texto_minusculas, language='spanish')
    tokens_limpios = []
    for palabra in tokens:
        if palabra.isalpha(): 
            if palabra not in stop_words_es: 
                palabra_raiz = stemmer.stem(palabra) 
                tokens_limpios.append(palabra_raiz)
    return " ".join(tokens_limpios)

def extraer_texto_y_paginas(archivo_pdf):
    # Esta función es para el clasificador TF-IDF.
    # Usa PyPDF2 y devuelve solo el texto completo.
    try:
        pdf_stream = io.BytesIO(archivo_pdf.read())
        reader = PdfReader(pdf_stream)
        texto_completo = ""
        for page in reader.pages:
            texto_pagina = page.extract_text()
            if texto_pagina:
                texto_completo += texto_pagina + "\n\n"
        # Devolvemos una tupla para ser compatible con la llamada en api_clasificar
        return texto_completo, None 
    except Exception as e:
        print(f"Error leyendo PDF con PyPDF2: {e}")
        return None, None

#--- Rutas de la Aplicación Web ---

@app.route('/')
def home():
    pdf_cargado = 'pdf_id' in session
    num_fragmentos = session.get('num_fragmentos', 0)
    return render_template('inspector.html', 
                           results=None, 
                           pdf_cargado=pdf_cargado, 
                           num_fragmentos=num_fragmentos)

#Búsqueda Semántica
@app.route('/inspector', methods=['POST'])
def search():
    if modelo is None:
        return "Error: El modelo de IA no está cargado.", 500

    query = request.form.get('query', '')
    file = request.files.get('pdf_file')

    #CASO 1: Se sube un NUEVO PDF
    if file and file.filename != '' and file.filename.endswith('.pdf'):
        try:
            print("Procesando nuevo PDF por página...")
            pdf_bytes = file.read()
        
            datos_fragmentos = extraer_y_dividir_por_pagina(pdf_bytes)
            
            if not datos_fragmentos:
                return render_template('inspector.html', query=query, error="No se pudo extraer texto o el PDF está vacío.")
            
            index_faiss, datos_originales = crear_indice_faiss(datos_fragmentos, modelo)
            
            pdf_id = str(uuid.uuid4())
            PDF_CACHE[pdf_id] = (index_faiss, datos_originales, pdf_bytes) 
            
            session['pdf_id'] = pdf_id
            session['num_fragmentos'] = len(datos_originales)
            
            resultados = buscar_en_indice(query, index_faiss, datos_originales, modelo, k=3)
            
            return render_template('inspector.html', 
                                   results=resultados,
                                   query=query, 
                                   num_fragmentos=len(datos_originales),
                                   pdf_cargado=True)
        
        except Exception as e:
            return render_template('inspector.html', error=f"Ocurrió un error: {e}")

    #CASO 2: El usuario ya subió un PDF y solo está consultando
    elif query != '' and 'pdf_id' in session:
        pdf_id = session['pdf_id']
        cached_data = PDF_CACHE.get(pdf_id)
        
        if cached_data:
            print(f"Usando índice en caché: {pdf_id}")
            index_faiss, datos_originales, _ = cached_data
            
            resultados = buscar_en_indice(query, index_faiss, datos_originales, modelo, k=3)
            
            return render_template('inspector.html', 
                                   results=resultados,
                                   query=query, 
                                   num_fragmentos=session['num_fragmentos'],
                                   pdf_cargado=True)
        else:
            session.pop('pdf_id', None)
            session.pop('num_fragmentos', None)
            return render_template('inspector.html', error="Tu sesión de PDF expiró. Por favor, sube el archivo de nuevo.")

    #CASO 3: Error
    else:
        return render_template('inspector.html', error="Debes subir un PDF y escribir una consulta.")

@app.route('/clear')
def clear_session():
    pdf_id = session.pop('pdf_id', None)
    session.pop('num_fragmentos', None)
    if pdf_id and pdf_id in PDF_CACHE:
        PDF_CACHE.pop(pdf_id, None) 
        print(f"Limpiando sesión y caché para {pdf_id}.")
    return redirect(url_for('home'))

@app.route('/get_pdf_viewer')
def get_pdf_viewer():
    if 'pdf_id' not in session:
        return "No hay PDF en la sesión.", 404
    pdf_id = session['pdf_id']
    cached_data = PDF_CACHE.get(pdf_id)
    if cached_data:
        pdf_bytes = cached_data[2]
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'inline; filename=documento.pdf'
        return response
    else:
        return "El PDF expiró de la caché.", 404

# --- Rutas del Clasificador ---

@app.route("/clasificador")
def clasificador_page():
    return render_template("clasificador.html")

@app.route("/api/clasificar", methods=["POST"])
def api_clasificar():
    """
    API para "Clasificador" usando SÓLO TF-IDF.
    Es mucho más rápido y preciso para palabras clave.
    """
    try:
        archivos_subidos = request.files.getlist('file')
        pregunta = request.form['query']
        
        if not archivos_subidos or archivos_subidos[0].filename == '':
            return jsonify({'error': 'No se seleccionaron archivos.'})
        
        #UMBRAL DE RELEVANCIA LÉXICA
        # Con TF-IDF, un puntaje bajo (ej. 0.05) ya puede ser relevante
        UMBRAL_LEXICAL = 0.05 
        
        #Preparar la Pregunta
        pregunta_limpia = preprocesar_texto(pregunta)
        
        lista_textos_completos = []
        lista_nombres_archivos = []

        #Extraer el texto de TODOS los archivos
        for archivo in archivos_subidos:
            if archivo and archivo.filename.endswith('.pdf'):
                texto_completo, _ = extraer_texto_y_paginas(archivo)
                if texto_completo:
                    lista_textos_completos.append(texto_completo)
                    lista_nombres_archivos.append(archivo.filename)

        if not lista_textos_completos:
            return jsonify([])

        #Preprocesar los textos
        corpus_limpio = [preprocesar_texto(texto) for texto in lista_textos_completos]

        #Aplicar TF-IDF
        vectorizer_tfidf = TfidfVectorizer()
        tfidf_doc_matriz = vectorizer_tfidf.fit_transform(corpus_limpio)
        tfidf_pregunta = vectorizer_tfidf.transform([pregunta_limpia])
        
        #Similitud de la pregunta contra TODOS los documentos a la vez
        similitudes_tfidf = cosine_similarity(tfidf_pregunta, tfidf_doc_matriz).flatten()

        #Filtrar y Recopilar Resultados
        lista_de_scores = []
        for i, score in enumerate(similitudes_tfidf):
            if score > UMBRAL_LEXICAL:
                lista_de_scores.append({
                    "documento": lista_nombres_archivos[i],
                    "similitud": float(score) 
                })

        #lista resultados relevantes
        resultados_ordenados = sorted(lista_de_scores, key=lambda x: x["similitud"], reverse=True)
        
        return jsonify(resultados_ordenados)
            
    except Exception as e:
        print(f"Error en /api/clasificar: {e}")
        return jsonify({'error': f'Ocurrió un error en el servidor: {str(e)}'}), 500
    
#--- Ejecutar la App ---
if __name__ == '__main__':
    app.run(debug=True)