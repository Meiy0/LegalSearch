import os
import io
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import numpy as np

#Configuración inicial
print("Inicializando Flask...")
app = Flask(__name__)

#Modelos y datos globales
print("Cargando modelo SentenceTransformer...")
try:
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Modelo cargado desde caché.")
except Exception as e:
    print(f"Error: {e}")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

#----- Funciones auxiliares -----

#Configuracion de nltk
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

#Toma el pdf y devuelve el texto
def extraer_texto(archivo_pdf):
    try:
        pdf_stream = io.BytesIO(archivo_pdf.read()) #leer desde memoria
        reader = PdfReader(pdf_stream)
        texto = ""
        for pagina in reader.pages:
            texto += pagina.extract_text() + "\n"
        return texto
    except Exception as e:
        print(f"Error leyendo PDF: {e}")
        return None

#Dividir texto en trozos de longitud fija
def segmentar_texto(texto_completo):
    #reemplazar saltos de linea
    texto_limpio = texto_completo.replace('\n', ' ').replace('\t', ' ')
    #quitar doble espacios
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
    
    #segmentar por oración
    oraciones = nltk.sent_tokenize(texto_limpio, language='spanish')
    
    #filtrar por longitud
    clausulas = [s for s in oraciones if len(s) > 20 and len(s) < 1000]
    
    return clausulas

#Motor de búsqueda semántica y lexical
def buscar_en_documento(texto_completo, pregunta):
    #segmentar en cláusulas
    clausulas = segmentar_texto(texto_completo)
    if not clausulas:
        return []
    
    #----- busqueda semantica -----
    #generar embeddings
    embedding_doc = modelo.encode(clausulas)
    embedding_pregunta = modelo.encode([pregunta])

    #calcular similitud del coseno
    similitudes_semanticas = cosine_similarity(embedding_pregunta, embedding_doc).flatten()

    #ordenar resultados en top 20
    indices_top_20 = similitudes_semanticas.argsort()[-20:][::-1]

    clausulas_candidatas = [clausulas[i] for i in indices_top_20]
    if not clausulas_candidatas:
        return []
    
    #----- reranking lexical -----
    #preprocesamiento
    clausulas_limpias = [preprocesar_texto(c) for c in clausulas_candidatas]
    pregunta_limpia = preprocesar_texto(pregunta)

    #aplicar TF-IDF para las 20 clausulas
    vectorizer_tfidf = TfidfVectorizer()
    doc_candidatos = vectorizer_tfidf.fit_transform(clausulas_limpias)

    pregunta_tfidf = vectorizer_tfidf.transform([pregunta_limpia])

    #calcular similitud
    similitudes_tfidf = cosine_similarity(pregunta_tfidf, doc_candidatos).flatten()

    #----- entregar resultados -----
    indices_top_5 = similitudes_tfidf.argsort()[-5:][::-1]

    #preparar resultados
    resultados = []
    for i in indices_top_5:
        index_og = indices_top_20[i]
        puntaje_semantico = similitudes_semanticas[index_og] #puntaje semantico original con orden TF-IDF

        if puntaje_semantico > 0.1: #umbral de confianza
            resultados.append({
                "similitud": float(puntaje_semantico),
                "clausula" : clausulas[index_og]
            })

    
    return resultados

#Endpoints de la API

#Home
@app.route("/")
def home():
    return render_template("search_demo.html")

#Búsqueda Semántica
@app.route("/search", methods=["POST"])
def buscar():
    #recibe archivo y pregunta
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se subió ningún archivo."})

        archivo = request.files['file']
        pregunta = request.form['query']

        if archivo.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo."})
        
        if archivo and archivo.filename.endswith('.pdf'):
            #extraer texto del pdf
            texto = extraer_texto(archivo)
            if texto is None:
                return jsonify({"error": "No se pudo leer el archivo PDF."})
            
            #conectar con motor de búsqueda 
            resultados = buscar_en_documento(texto, pregunta)
            
            #devolver resultados
            return jsonify(resultados)
        else:
            return jsonify({"error": "Por favor, sube un archivo .pdf"})
    except Exception as e:
        print(f"Error en /buscar: {e}")
        return jsonify({'error': f'Ocurrió un error en el servidor: {str(e)}'}), 500

#Ejecución de la aplicación
if __name__ == "__main__":
    app.run(debug=True, port=5000)