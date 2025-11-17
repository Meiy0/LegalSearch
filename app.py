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
import unicodedata
from rapidfuzz import fuzz


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
#Cargar terminos legales
def cargar_terminos_desde_archivo(filename="terminos_legales.txt"):
    """
    Carga los términos legales desde un archivo .txt.
    Ignora líneas vacías y quita espacios en blanco.
    """
    terminos = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for linea in f:
                termino_limpio = linea.strip() 
                if termino_limpio:
                    terminos.append(termino_limpio)
        print(f"✅ Se cargaron {len(terminos)} términos legales desde {filename}")
        
    except FileNotFoundError:
        print(f"⚠️ Error: No se encontró el archivo {filename}.")
        print("Usando una lista de respaldo por defecto.")
        terminos = ["contrato", "cláusula", "error"] # Lista de emergencia
        
    except Exception as e:
        print(f"Error inesperado leyendo el archivo {filename}: {e}")
        terminos = []
        
    return terminos
LISTA_TERMINOS_LEGALES = cargar_terminos_desde_archivo("terminos_legales.txt")

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

#Extrae el pdf completo
def extraer_texto_completo(stream_de_bytes):
    texto_completo = ""
    with fitz.open(stream=stream_de_bytes, filetype="pdf") as doc:
        for pagina in doc:
            texto_completo += pagina.get_text() + "\n"
    return texto_completo

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
    try:
        pdf_stream = io.BytesIO(archivo_pdf.read())
        reader = PdfReader(pdf_stream)
        texto_completo = ""
        for page in reader.pages:
            texto_pagina = page.extract_text()
            if texto_pagina:
                texto_completo += texto_pagina + "\n\n"

        return texto_completo, None 
    except Exception as e:
        print(f"Error leyendo PDF con PyPDF2: {e}")
        return None, None

def normalizar_texto(texto):
    if not texto:
        return ""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = texto.replace("\n", " ")
    texto = texto.replace("\t", " ")
    return " ".join(texto.split())

def segmentar_texto(texto):
    partes = re.split(r"\n\s*\n", texto)
    return [p.strip() for p in partes if len(p.strip()) > 100]



#--- Rutas de la Aplicación Web ---

@app.route('/')
def home():
    pdf_cargado = 'pdf_id' in session
    num_fragmentos = session.get('num_fragmentos', 0)
    return render_template('inspector.html', 
                           results=None, 
                           pdf_cargado=pdf_cargado, 
                           num_fragmentos=num_fragmentos)

#-- Rutas de la Búsqueda Contextual ---
@app.route('/inspector', methods=['POST'])
def search():
    if modelo is None:
        return "Error: El modelo de IA no está cargado.", 500

    query = request.form.get('query', '')
    file = request.files.get('pdf_file')

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

    try:
        archivos_subidos = request.files.getlist('file')
        query_raw = request.form.get('query', '').strip()

        if not archivos_subidos or archivos_subidos[0].filename == '':
            return jsonify({'error': 'No se seleccionaron archivos.'}), 400

        if query_raw == '':
            return jsonify({'error': 'Debes escribir una palabra o frase a buscar.'}), 400

        query = normalizar_texto(query_raw)

        resultados = []

        for archivo in archivos_subidos:

            if archivo and archivo.filename.endswith('.pdf'):

                archivo.seek(0)
                pdf_stream = io.BytesIO(archivo.read())
                reader = PdfReader(pdf_stream)

                texto_completo = ""

                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        texto_completo += " " + normalizar_texto(txt)

                coincidencia_exacta = query in texto_completo
                palabras = query.split()

                if len(palabras) == 1:
                    score_fuzzy = fuzz.partial_ratio(query, texto_completo)
                else:
                    score_fuzzy = fuzz.token_set_ratio(query, texto_completo)

                fuzzy_match = score_fuzzy >= 80

                resultados.append({
                    "documento": archivo.filename,
                    "coincidencia_exacta": coincidencia_exacta,
                    "fuzzy": fuzzy_match,
                    "score": float(score_fuzzy)
                })

        resultados = sorted(
            resultados,
            key=lambda x: (x["coincidencia_exacta"], x["fuzzy"], x["score"]),
            reverse=True
        )

        return jsonify(resultados)

    except Exception as e:
        print("Error en búsqueda fuzzy:", e)
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500


#--- Rutal del Destacador ---
@app.route("/destacador")
def destacador_page():
    return render_template("destacador.html")

@app.route("/api/destacar_pdf", methods=["POST"])
def api_destacar_pdf():
    """
    API Corregida: Recibe 1 PDF y lo devuelve destacando en amarillo
    los términos de LISTA_TERMINOS_LEGALES, ignorando mayúsculas
    y manejando espacios múltiples o saltos de línea.
    
    Esta versión incluye una corrección para evitar la superposición
    de destacados (que los hacía ver más oscuros).
    """
    
    if not LISTA_TERMINOS_LEGALES:
        return "Error: No se cargó la lista de términos legales en el servidor.", 500
    
    try:
        archivo = request.files.get('pdf_file')
        
        if not archivo or archivo.filename == '' or not archivo.filename.endswith('.pdf'):
            return "No se seleccionó un archivo PDF válido.", 400

        pdf_bytes = archivo.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        print(f"Buscando {len(LISTA_TERMINOS_LEGALES)} términos legales con regex...")

        AMARILLO_DESTACADOR = (0.95, 0.95, 0.2) 


        for pagina in doc:
            
            areas_ya_destacadas = [] 

            anotaciones_viejas = pagina.annots() 
            for annot in anotaciones_viejas:
                pagina.delete_annot(annot)
            

            texto_pagina = pagina.get_text("text")

            for termino in LISTA_TERMINOS_LEGALES:
                
                termino_escaped = re.escape(termino)
                
                patron_regex = termino_escaped.replace(r'\ ', r'\s+')
                
                matches = re.finditer(patron_regex, texto_pagina, flags=re.IGNORECASE)
                
                for match in matches:

                    texto_encontrado = match.group(0)
                    
                    areas_encontradas = pagina.search_for(texto_encontrado, quads=True)
                    
                    for quad in areas_encontradas:
                        
                        rect_actual = quad.rect

                        hay_superposicion = False
                        for rect_existente in areas_ya_destacadas:

                            if rect_actual.intersects(rect_existente):
                                hay_superposicion = True
                                break
                        
                        if not hay_superposicion:
                            annot = pagina.add_highlight_annot(quad)
                            
                            annot.set_colors(stroke=AMARILLO_DESTACADOR)
                            annot.update()
                            
                            areas_ya_destacadas.append(rect_actual) 
                            
        output_bytes = doc.tobytes()
        doc.close()

        response = make_response(output_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        
        nombre_archivo = f"destacado_{archivo.filename}"
        response.headers['Content-Disposition'] = f'attachment; filename="{nombre_archivo}"'
        
        return response

    except Exception as e:
        print(f"Error general en /api/destacar_pdf: {e}")
        return f"Ocurrió un error en el servidor: {str(e)}", 500


#--- Ejecutar la App ---
if __name__ == '__main__':
    app.run(debug=True)