#Usar la base de datos
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
  token="AstraCS:gJIBilICvSraBwzdFLZpbdlq:5071edfb8291d493f575622de1551d497c817ee48683979e58578110b8f30b86",
  api_endpoint="https://3be10551-48af-4652-b761-5c0e4b3d25dd-us-east-1.apps.astra.datastax.com")

print(f"Connected to Astra DB: {db.get_collections()}")
#Create a collection
collection = db.collection("vector_movies")


def buscar_en_coleccion(nombre_pelicula):
    vector_busqueda = get_embedding(nombre_pelicula)
    max_records = 10

    resultados = collection.vector_find(vector=vector_busqueda, limit=max_records)
    return resultados

def buscar_peliculas_similares(b):
    with output:
        clear_output()
        #Obtener el promp del usuario
        prompt= input_text.value
        if prompt:
            resultados = buscar_en_coleccion(prompt)
            #Mostrar los resultados
            print("Películas similares encontradas:")
            for pelicula in resultados:
                print(pelicula.get("text"))
        else:
            print("Por favor introduce un nombre de pelicula")

#Imprimir resultados
nombre_pelicula = "Shrek"
resultados = buscar_en_coleccion(nombre_pelicula)
for result in resultados:
    print(result.get("text"))