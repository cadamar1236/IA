from astrapy.db import AstraDB
import langchain
from langchain_openai import OpenAIEmbeddings
import json
import os
from dotenv import load_dotenv 
# Cargar variables de entorno
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
# Configurar el motor de OpenAI
engine = "gpt-4"
embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
def get_embedding(text):
    query_result = embeddings.embed_query(text)
    print(query_result)
    return query_result

# Initialization
db = AstraDB(
  token="AstraCS:TMDZuFSwIXefNCPRjgtScxAi:7a34b391a75e6b6cb2f6230784b939609c12f3d7ce28bf5ccfb1200645121b34",
  api_endpoint="https://3be10551-48af-4652-b761-5c0e4b3d25dd-us-east-1.apps.astra.datastax.com")

print(f"Connected to Astra DB: {db.get_collections()}")
#Create a collection
collection = db.create_collection("vector_movies", dimension=3072, metric = "cosine")
#Procesar cada línea y obtener los embeddings
#LEeer el documento de texto
with open('IMDB-Movie-Data.csv', 'r') as file:
          lines = file.readlines()

documents = []

for index, line in enumerate(lines):
    # Extraer el texto de cada línea

    # Obtener el embedding para el texto
    vector = get_embedding(line)

    # Crear el documento
    document = {
        "_id": str(index + 1),
        "text": line,
        "$vector": vector
    }

    # Insertar el documento en la base de datos
    # Reemplaza 'collection' con tu objeto de colección de la base de datos
    # res = collection.insert_one(document)  # Usamos insert_one para insertar un solo documento
    res = collection.upsert(document)  # Aqui usamos upsert si existe se actualiza si no, se crea
    # Añadir al documento JSON
    documents.append(document)

# Guardar los documentos en un archivo JSON
with open('documentos.json', 'w') as file:
    json.dump(documents, file)
print(documents)