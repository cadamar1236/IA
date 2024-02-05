from openai import OpenAI
from dotenv import load_dotenv
import os
# Cargar variables de entorno
load_dotenv()
# Configurar el motor de OpenAI
engine = "gpt-4-1106-preview"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
    model=engine,
    messages=messages,
    temperature=0.3, # esta es la creatividad del modelo
    )
    return response.choices[0].message.content

curso = f"""
1. Introducción
-Entorno de desarrollo
-Intérprete
-Hola Mundo
-Python2 vs Python3
2. Sintaxis de Python
-Tipos de datos básicos
-Tipos compuestos, secuencias
-Operadores
-Condicionales
-Control de flujo
-Bucles
-Funciones
3. Strings y salida por consola
-Funciones básicas de strings
4. Secuencias, listas y diccionarios
-Operaciones básicas
-Búsqueda, ordenación
-Iteradores
-Generadores
-Lambdas
5. Modularidad
-Módulos standard
-Nuestros propios módulos
-Packages
6. Clases
-Programación Orientada a Objetos
-Herencia, Encapsulación, polimorfismo
-Programación de clases
7. Ficheros
-Lectura y escritura en ficheros
-Formatos
8. Excepciones
-Tratamiento de errores
-Try/except/else/finally
-Programación de excepciones propias
Audiencia:
Curso dirigido a principiantes en programación y profesionales que \
deseen adquirir habilidades en el desarrollo de software utilizando Python.
\También puede ser interesante para aquellos que deseen comenzar su viaje \
en la ciencia de datos, ya que Python es una herramienta esencial en este campo.
Prerrequisitos:
Los alumnos necesitarán tener conocimientos básicos de programación.
Objetivos:
Este curso tiene como objetivo proporcionar a los participantes una base sólida \
en programación utilizando Python como lenguaje principal.
Al finalizar el curso, los participantes estarán capacitados para escribir \
programas simples utilizando los conceptos y técnicas aprendidos, comprender \
y modificar código existente, así como abordar problemas y desafíos de \
programación utilizando Python.
Código: OPSPY02
Metodología: ILT
Duración: 30 Horas
Habilidades: Desarrollo de Aplicaciones
"""
prompt = f"""
Tu tarea es ayudar a un equipo de marketing a \
crear una descripción para un sitio web de ventas \
de un producto, basándote en una ficha técnica. 
Escribe una descripción del producto con base en la \
información proporcionada en la ficha de curso 
del curso delimitado por triple acento grave
No incluyas datos que ya aparecen en la ficha como la
duración o el código.
Usa como máximo 80 palabras. Enfócate en los \
conceptos relativos a el futuro que tienen los conocimientos,\
la utilidad y el desarrollo profesional.
Ficha de curso: ```{curso}```
"""
response = get_completion(prompt)
print(response)
