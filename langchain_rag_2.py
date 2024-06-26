import os
from dotenv import load_dotenv

#from openai import OpenAI
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

db = Chroma(persist_directory=persistent_dir,
            embedding_function=embeddings)

query = "What is the IC SCT3604 does?"

# Retrieve the most similar documents based on query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.4}
)
related_docs = retriever.invoke(query)

# Display the retrieved result
print("Relevent documents:")
for i,doc in enumerate(related_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

