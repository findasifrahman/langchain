import os
from dotenv import load_dotenv

#from openai import OpenAI
from langchain_openai import ChatOpenAI
import re,time
from firecrawl import FirecrawlApp
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")#, api_key=OPENAI_API_KEY)

message = [
    SystemMessage(content="SOLVE given MATH PROBLEM"),
    HumanMessage(content="What is 4+2?"),
]

# INVOKE THE MODEL
result = model.invoke(message)

print(result)