# text splitting example using metadata

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
db_dir = os.path.join(current_dir, "db")
pdf_path = os.path.join(current_dir, "data")

if not os.path.exists(persistent_dir):
    print("persistent directory does not exist, creating it")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"file {pdf_path} does not exist")
    
    pdf_path = os.path.join(current_dir, "data")
    document = []
    for file in os.listdir(pdf_path):
        pdf_path = os.path.join(current_dir, "data")
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_path, file)
            loader = PyPDFLoader(pdf_path)
            #document.extend(loader.load())
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata = {"source": file}
                document.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    docs = text_splitter.split_documents(document)

    print("\nNumber of chunks created:", len(docs))

    # create embeddings for each chunk
    print("\nCreating embeddings for each chunk\n")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("Embeddings created\n")

    # create a vector store
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir
    )
    print("Vector store created\n")

else:
    print("persistent directory exists")