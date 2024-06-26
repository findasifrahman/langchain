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
from langchain.schema.runnable import RunnableLambda,RunnableParallel
from langchain_community.callbacks import get_openai_callback


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")#ChatOpenAI(model="gpt-4o")#ChatOpenAI(model="gpt-3.5-turbo")#, api_key=OPENAI_API_KEY)



prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an Expert Company Reviewer"),
    ("human", "List the main features of the company {company_name}")
])

# Define pros
def pros(features):
    pros_template = ChatPromptTemplate.from_messages([
        ("system", "You are an Expert Company Reviewer"),
        ("human", "Given these features: {features}, list the pros of the company")

    ])
    return pros_template.format_prompt(features=features)

def cons(features):
    cons_template = ChatPromptTemplate.from_messages([
        ("system", "You are an Expert company Reviewer"),
        ("human", "Given these features: {features}, list the cons of the company")
    ])
    return cons_template.format_prompt(features=features)

def combine_pros_cons(pros, cons):
    return f"Pros: \n{pros}\n\nCons: \n{cons}"

pros_branch = (
    RunnableLambda(lambda x: pros(x)) | model | StrOutputParser() 
)

cons_branch = (
    RunnableLambda(lambda x: cons(x)) | model | StrOutputParser()
)

chain = (
    prompt_template 
    | model 
    | StrOutputParser() 
    | RunnableParallel(branches={"pros": pros_branch,"cons": cons_branch})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"],x["branches"]["cons"] ))
)

with get_openai_callback() as cb:
    response = chain.invoke({"company_name": "MalishaEdu"})
    print(response)

print(f"Total Tokens: {cb.total_tokens}")