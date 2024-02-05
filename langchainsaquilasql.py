from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv 
# Cargar variables de entorno
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
# Inicializamos db
db = SQLDatabase.from_uri("sqlite:///sqlite-sakila.db")

llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4", temperature=0)

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
resultado = agent_executor.invoke(
    "Dime las columnas que tiene el dataset. Dame tu respuesta en castellano"
)
print(resultado.get('output'))