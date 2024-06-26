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
from langchain.schema.runnable import RunnableLambda,RunnableParallel,RunnableBranch
from langchain_community.callbacks import get_openai_callback


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")#ChatOpenAI(model="gpt-4o")#ChatOpenAI(model="gpt-3.5-turbo")#, api_key=OPENAI_API_KEY)

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a thank you note for this positive feedback: {feedback}")
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a response addressing this negative feedback: {feedback}")
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}")
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a response to escalate this feedback to a human agent: {feedback}")
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}")

    ]
)

# define the runnable branches for handling feedback

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
    
)

classification_chain = classification_template | model | StrOutputParser()
# combine classification and response generation into the chain
chain = classification_chain | branches

# run the chain with an example
# Good review - "I love this product, it's the best thing ever!"
# Bad review - "This product is terrible, I want a refund!"
# Neutral review - "This product is okay, I guess."
# Escalate review - "I am having trouble with this product, please help!"
# default - "I am not sure about the product yesy.Can you tell me more about it?"

review =  "The Product was unbelievably bad, It sucks."
result = chain.invoke({"feedback": review})

print(result)