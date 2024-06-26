import os
from dotenv import load_dotenv

#from openai import OpenAI
from langchain_openai import ChatOpenAI
import re,time
from firecrawl import FirecrawlApp
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_firestore import FirestoreChatMessageHistory
from google.auth import compute_engine
from google.cloud import firestore

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
PROJECT_ID = "langchain-chat-64439"#os.getenv('FIRESTORE_PROJECT_ID')
SESSION_ID = "user_session_new"#os.getenv('FIRESTORE_SESSION_ID')
COLLECTION_NAME = "chat_history"#os.getenv('FIRESTORE_COLLECTION_NAME')

load_dotenv()
print(PROJECT_ID)
print(SESSION_ID)

'''
    Create a new collection in Firestore
    Create a new document in the collection
    Create a new field in the document
    install google clud cli on my computer
        - https://cloud.google.com/sdk/docs/install
        - authenticate with google cloud with google account
            https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
        - set your project to the new firestore project you created
    Enable the firestore API in google cloud console
        - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
'''


print("starting firestore app")

model = ChatOpenAI(model="gpt-3.5-turbo")#, api_key=OPENAI_API_KEY)

client = firestore.Client(project=PROJECT_ID)

chat_history = FirestoreChatMessageHistory(session_id=SESSION_ID, collection=COLLECTION_NAME, client=client)

print("Current Chat History: ", chat_history)

while True:
    human_input = input("You: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input) # add human message to chat history

    # AI response
    ai_response = model.invoke(chat_history.messages)

    chat_history.add_ai_message(ai_response.content)# add AI message to chat history

    print(f"AI: {ai_response.content}")