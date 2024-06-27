# explore which text splitter is useful and when to use
# character text splitter or sentence text splitter

import os
from dotenv import load_dotenv

#from openai import OpenAI
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter, TokenTextSplitter, TextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader
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



    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    def create_vector_store(docs, store_name):
        # create embeddings for each chunk
        persistent_dir = os.path.join(db_dir, store_name)
        if not os.path.exists(persistent_dir):
            print(f"Creating vector store {store_name}")
            db = Chroma.from_documents(
                docs, embeddings, persist_directory=persistent_dir
            )
            print(f"Vector store {store_name} created")
        else:
            print(f"Vector store {store_name} exists")


    # 1. character text splitter
    # SPLITS text into chunks of fixed size on a specific character
    # Useful for consistent chumk sizes regardless of the content and structure
    text_splitter = CharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    docs = text_splitter.split_documents(document)
    create_vector_store(docs, "chroma_db_char")

    # 2. sentence based splitter
    # SPLITS text into chunks based on sentences, ensuring chunks end at sentence boundaries
    # Ideal for maintaining the semantic integrity of the text
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=2000)
    docs = text_splitter.split_documents(document)
    create_vector_store(docs, "chroma_db_sent")

    # 3. Token based splitter
    #  SPLITS TEXT INTO CHUMKS BASED ON TOKENS (WORD OR SUBWORD)
    # balance between maintaining coherence and adhering to char
    print("\n Using token based splitter")
    token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
    docs = token_splitter.split_documents(document)
    create_vector_store(docs, "chroma_db_token")

    # 4. Recursive Character-based Splitting . most people uses for docs
    # Attempts to split text at natural boundaries( sentences. paragaraphs) within char limit
    # Balances between maintain coherence and adhering to char limit
    print("\n Using recursive character based splitter")
    token_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    rec_char_docs = token_splitter.split_documents(document)
    create_vector_store(rec_char_docs, "chroma_db_rec_char")

    # 5. Custom splitting
    # Allows for custom splitting based on a function
    # Useful for domain-specific splitting requirements
    '''
    print("\n Using custom splitter")
    class CustomTextSplitterTest(TextSplitter):
        def split_text(self, text):
            # custom splitting logic
            return text.split("\n\n") # Example split by paragaraph

    custom_splitter = CustomTextSplitterTest()
    custom_docs = custom_splitter.split_text(document)
    create_vector_store(custom_docs, "chroma_db_custom")
    '''

# Function to query a vector store
def query_vector_store(store_name, query):
    persistent_dir = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_dir):
        print(f"Querying vector store {store_name}")
        db = Chroma(persist_directory=persistent_dir,
                    embedding_function=embeddings)
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.2}
        )
        related_docs = retriever.invoke(query)
        print(f"\n-- Relevent documents in {store_name} --")
        for i,doc in enumerate(related_docs, 1):
            print(f"Document {i}: \n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

    else:
        print(f"Vector store {store_name} does not exist")

    

query = "what is the total pin of sct3604 ic?"

# Query the different vector stores
query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
#query_vector_store("chroma_db_custom", query)

 

  