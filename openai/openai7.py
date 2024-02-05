from openai import OpenAI
from dotenv import load_dotenv
import os
# Cargar variables de entorno
load_dotenv()
# Configurar el motor de OpenAI
engine = "gpt-4-1106-preview"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_completion(prompt):
    completion = client.chat.completions.create(
    model=engine,
    messages=[
    {"role": "user", "content": f"{prompt}"}
    ]
    )
    return completion

prompt = f"""
Determina si la solución del alumno es correcta o no. Hazlo primero sin tener en cuenta la solución del alumno y luego compara con la del alumno
para ver si coinciden y si es o no correcta.
Pregunta:
```
Estoy construyendo una instalación de energía solar y necesito ayuda para calcular los aspectos financieros.
El terreno cuesta $100 / pie cuadrado
Puedo comprar paneles solares por $250 / pie cuadrado
Negocié un contrato de mantenimiento que me costará un fijo de $100k por año, y un adicional de $10 / pie cuadrado
¿Cuál es el costo total para el primer año de operaciones en función del número de pies cuadrados?
``` 
Solución del alumno:
```
Supongamos que x es el tamaño de la instalación en pies cuadrados.
Costos:
Costo del terreno: 100x
Costo de los paneles solares: 250x
Costo de mantenimiento: 100,000 + 100x
Costo total: 100x + 250x + 100,000 + 100x = 450x + 100,000
```
"""
response = get_completion(prompt)
print(response.choices[0].message.content)