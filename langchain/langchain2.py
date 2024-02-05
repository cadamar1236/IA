from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
prompt = ChatPromptTemplate.from_template(
"Tell me a short joke about {topic}"
)
api_key=os.getenv("OPENAI_API_KEY")
output_parser = StrOutputParser()
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=1)
chain = (
{"topic": RunnablePassthrough()}
| prompt
| model
| output_parser
)
response = chain.invoke("ice cream")
print(response)
