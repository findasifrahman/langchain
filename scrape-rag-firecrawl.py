# Use langchain and firecrawl to scrap web and chat with it


import os
from dotenv import load_dotenv

#from openai import OpenAI
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, FireCrawlLoader
from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chroma_db_firecrawl_scrap")

def create_vector_store():
    """Crawl the website, split the content, create Embedding, and persist the vector store """
    # define firecrawl
    api_key_firecrawl = os.getenv('FIRECRAWL_API_KEY')

    # STEP 1: CRAWL USING FIRECRAWL
    url = "https://malishaedu.com"

    # create a loader for web content
    loader = FireCrawlLoader(api_key=api_key_firecrawl, url=url,mode="scrape")
    documents = loader.load()
    print("finished crawl")

    # convert metadata values into string if they are list
    for doc in documents:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str,value))
    
    # step 2: split the web document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)#RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    # Display info about the split document
    print("\nChunk Information\n")
    print(f"Number of chunks: {len(docs)}")
    print(f"Sample chunk:\n {docs[0].page_content}\n")

    # create embeddings for each chunk
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # step 4: Create and persist the vector store with the embeddings
    # Chroma stores the embeddings for search


    print("persistent directory does not exist, creating it")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir
    )
    print("Persistent directory created\n")

if not os.path.exists(persistent_dir):
    create_vector_store()
else:
    print("Vector store already exists")

    
# step 5: Retrieve the most similar documents based on query

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
db = Chroma(persist_directory=persistent_dir,
            embedding_function=embeddings)

def query_vector_store(query):
    """
    Query the vector store with specified question.
    """

    retriver = db.as_retriever(
        search_type='similarity',#"similarity_score_threshold",
        search_kwargs={"k": 3}#{"k": 1, "score_threshold": 0.3}
    )



    relavent_docs = retriver.invoke(query)

    # Display the relavent results with metadata
    print("\nRelevent documents:")
    for i,doc in enumerate(relavent_docs, 1):
        print(f"Document {i}: \n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# Define the users question
query = "What is the name of malishaEdu chairman?"

query_vector_store(query)