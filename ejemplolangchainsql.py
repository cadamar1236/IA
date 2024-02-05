from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
import os
from dotenv import load_dotenv 
# Cargar variables de entorno
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

# Inicializamos db
db = SQLDatabase.from_uri("sqlite:///chinook.db")
# Pull down prompt
prompt = hub.pull("rlm/text-to-sql")
# Initialize model
model = ChatOpenAI(openai_api_key=api_key)

# Crear chain con LangChain Expression Language
inputs = {
    "table_info": lambda x: db.get_table_info(),
    "input": lambda x: x["question"],
    "few_shot_examples": lambda x: "",
    "dialect": lambda x: db.dialect,
}
sql_response = (
    inputs
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# Call with a given question
sql_response.invoke({"question": "Enumera las ventas totales por país. ¿Los clientes de qué país gastaron más?"})
