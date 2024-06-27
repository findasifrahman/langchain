
import os
from dotenv import load_dotenv

#from openai import OpenAI
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chroma_db_token")
pdf_path = os.path.join(current_dir, "data")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

db = Chroma(persist_directory=persistent_dir,
            embedding_function=embeddings)

query = "what is the total number of pin in sct3604 ic?"

# Retrieve the most similar documents based on query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.3}
)

retriever_docs = retriever.invoke(query)

# Display the relavent results with metadata
print("Relevent documents:")
for i,doc in enumerate(retriever_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")

# combine the query and relavent docs to generate a response
combine_input = (
    "Here are some documents that might help you with your question:"
    + query
    + "\n\nRelevent documents:\n"
    + "\n\n".join([doc.page_content for doc in retriever_docs])
    + "\n\nPlease provide an answer based only on the provided documents"

)

model = ChatOpenAI(model="gpt-4o")

message = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combine_input)
]

result = model.invoke(message)

# Display the full result
print("Response:")
print("Full result")
print(result)
print("content only:")
print(result.content)