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
file_path = os.path.join(current_dir, "data", "CMX940_datasheet.pdf")
persistent_dir = os.path.join(current_dir, "db", "chroma_db")

# check if the chroma directory already exists
if not os.path.exists(persistent_dir):
    print("persistent directory does not exist, creating it")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file {file_path} does not exist")
    
    # Read the content of file
    pdf_path = os.path.join(current_dir, "data")
    document = []
    for file in os.listdir(pdf_path):
        pdf_path = os.path.join(current_dir, "data")
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_path, file)
            loader = PyPDFLoader(pdf_path)
            document.extend(loader.load())

    #loader = PyPDFLoader(file_path)#TextLoader(file_path)
    #document = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=10)
    docs = text_splitter.split_documents(document)

    # display info about the split document
    print(f"Sample chunk:\n {docs[0].page_content}\n")

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

