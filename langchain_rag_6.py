# check different text embedding in OpenAIEmbedding and hugginf face BGE embedding
import os
from dotenv import load_dotenv

#from openai import OpenAI
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chroma_db")
pdf_path = os.path.join(current_dir, "data")
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



def create_vector_store(docs, embeddings, store_name):
    # create embeddings for each chunk
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):
        print(f"Creating vector store {store_name}")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_dir
        )
        print(f"Vector store {store_name} created")
    else:
        print(f"Vector store {store_name} exists")

# 4. Recursive Character-based Splitting . most people uses for docs
# Attempts to split text at natural boundaries( sentences. paragaraphs) within char limit
# Balances between maintain coherence and adhering to char limit
print("\n Using recursive character based splitter")
token_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
rec_char_docs = token_splitter.split_documents(document)


# 1. Open AI embedding 
# useful for general purpose with high accuracy
# The cost will depend on API usage
print("\n Using OpenAI Embeddings")
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)
create_vector_store(rec_char_docs,openai_embeddings, "chroma_db_openai")

# 2. Hugging Face BGE Embeddings
# uses models from hugging face library
# Ideal for levaraging a wide varity of model with different tasks

print("\n Using Hugging Face BGE Embeddings")
hugging_face_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
create_vector_store(rec_char_docs,hugging_face_embeddings, "chroma_db_hugging_face")

def query_vector_store(store_name, query, embedding_function):
    if os.path.exists(os.path.join(db_dir, store_name)):
        db = Chroma(persist_directory=os.path.join(db_dir, store_name),
                    embedding_function=embedding_function)


        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 2, "score_threshold": 0.3}
        )
        related_docs = retriever.invoke(query)
        print(f"Relevent documents for query '{query}' in store {store_name}:")
        for i,doc in enumerate(related_docs, 1):
            print(f"Document {i}: \n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata['source']}\n")

    else:
        print(f"Vector store {store_name} does not exist")

query = "what is the total number of pin in sct3604 ic?"

# Query the different vector stores
query_vector_store("chroma_db_openai", query, openai_embeddings)
query_vector_store("chroma_db_hugging_face", query, hugging_face_embeddings)

print("\nDone")