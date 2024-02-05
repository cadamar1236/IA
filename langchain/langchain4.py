from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import sys
import os
from dotenv import load_dotenv 
# Cargar variables de entorno
load_dotenv()
# Configurar el motor de OpenAI
engine = "gpt-3.5-turbo"
api_key=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.0, model=engine, openai_api_key=api_key)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=False # Si ponemos verbose=True devuelve datos de la plantilla
)
conversation.predict(input="Hola, mi nombre es Carlos")
conversation.predict(input="¿Cuál es la capital de Honduras?")
conversation.predict(input="¿Cuál es mi nombre?")
print(memory.buffer)
