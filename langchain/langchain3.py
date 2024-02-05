from langchain_openai import ChatOpenAI
import asyncio
import sys
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar el motor de OpenAI
engine = "gpt-3.5-turbo"
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(api_key=api_key)
chunks = []

async def chatear():
    async for chunk in model.astream("hello. tell me something about yourself"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)

# Llamar a la función principal usando asyncio.run()
if __name__ == "__main__":
    asyncio.run(chatear())
