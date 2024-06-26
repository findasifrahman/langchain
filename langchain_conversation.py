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

chat_histoy = []

systemMessage = SystemMessage(content="You are a Helpful assistant")
chat_histoy.append(systemMessage)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    chat_histoy.append(HumanMessage(content=query)) # add human message to chat history

    # AI response
    result = model.invoke(chat_histoy)
    response = result.content

    chat_histoy.append(AIMessage(content=response)) # add AI message to chat history

    print("AI: ", response)

print("Message History: ", chat_histoy)