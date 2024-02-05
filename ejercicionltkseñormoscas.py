import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk

# Descarga recursos adicionales (solo necesitas hacerlo una vez)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Texto proporcionado
texto = "Ralph se sentó en la arena caliente, tocando la concha, sonriendo y asintiendo con la cabeza a los gritos de admiración. A su alrededor, los niños comenzaron a asentarse y a prestar atención. Era como si hubiesen oído por primera vez la brillante idea de la democracia."

# Tokenización de oraciones
oraciones = sent_tokenize(texto)
print("Oraciones:")
print(oraciones)

# Tokenización de palabras
palabras = word_tokenize(texto)
print("\nPalabras:")
print(palabras)

# Etiquetado de Partes del Discurso (POS Tagging)
pos_tags = pos_tag(palabras)
print("\nEtiquetado POS:")
print(pos_tags)

# Lematización
lemmatizador = WordNetLemmatizer()
lemas = [lemmatizador.lemmatize(palabra) for palabra in palabras]
print("\nLematización:")
print(lemas)

# Análisis de Sentimientos
analisador_sentimientos = SentimentIntensityAnalyzer()
polaridad = analisador_sentimientos.polarity_scores(texto)
print("\nAnálisis de Sentimientos:")
print(polaridad)

# Extracción de Entidades Nombradas (NER)
entidades_nombradas = ne_chunk(pos_tags)
print("\nEntidades Nombradas:")
print(entidades_nombradas)

#Ahora lo hacemos con spacy
import spacy

# Cargar el modelo en español de spaCy
nlp = spacy.load('es_core_news_sm')

# Texto de ejemplo en español
texto = """Ralph se sentó en la arena caliente, tocando la concha, sonriendo y asintiendo con la cabeza a los gritos de admiración. 
A su alrededor, los niños comenzaron a asentarse y a prestar atención. 
Era como si hubiesen oído por primera vez la brillante idea de la democracia."""

# Procesar el texto con spaCy
doc = nlp(texto)

# Tokenización y Etiquetado POS
oraciones = list(doc.sents)
palabras = [word.text for word in doc]
etiquetas_pos = [(word.text, word.pos_) for word in doc]

# Lematización
lemas = [word.lemma_ for word in doc]

# Análisis de Sentimientos
# NLTK SentimentIntensityAnalyzer no es óptimo para español, se puede usar otra herramienta o traducir el texto a inglés

# Extracción de Entidades Nombradas (NER)
entidades = [(ent.text, ent.label_) for ent in doc.ents]

# Imprimir resultados
print("Oraciones tokenizadas:")
for oracion in oraciones:
    print(oracion.text)
print("\nPalabras tokenizadas:")
print(palabras)
print("\nEtiquetas POS:")
print(etiquetas_pos)
print("\nLemas:")
print(lemas)
print("\nEntidades Nombradas:")
print(entidades)