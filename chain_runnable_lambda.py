import os
from dotenv import load_dotenv

#from openai import OpenAI
from langchain_openai import ChatOpenAI
import re,time
from firecrawl import FirecrawlApp
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_community.callbacks import get_openai_callback


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")#ChatOpenAI(model="gpt-4o")#ChatOpenAI(model="gpt-3.5-turbo")#, api_key=OPENAI_API_KEY)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells dirty jokes about {topic}"),
    ("human", "Tell me {joke_count} jokes?")
])

Uppercase = RunnableLambda(lambda x: x.upper())

chain = prompt_template | model | StrOutputParser() | Uppercase

with get_openai_callback() as cb:
    response = chain.invoke({"topic": "Elon Musk", "joke_count": 2})
    print(response)

#result = chain.invoke({"topic": "Elon Musk", "joke_count": 2})

print(f"Total Tokens: {cb.total_tokens}")