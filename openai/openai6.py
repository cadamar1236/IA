from openai import OpenAI
from dotenv import load_dotenv
import os
# Cargar variables de entorno
load_dotenv()
# Configurar el motor de OpenAI
engine = "gpt-3.5-turbo"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def get_completion(prompt):
    completion = client.chat.completions.create(
    model=engine,
    messages=[
    {"role": "system", "content": "Eres un experto en cocina con más de 10 años de experiencia, \
    tu conocimiento se extiende por múltipleas aspectos de la alimentación y dispones de \
    numerosas certificaciones en este campo. Además eres experto en cocina mediterránea."},
    {"role": "user", "content": f"{prompt}"}
    ]
    )
    return completion
prompt = (
f"Explica cómo hacer un menú que contenga todos los ingredientes típicos de España."
)
respuesta = get_completion(prompt)
print(respuesta.choices[0].message.content)
