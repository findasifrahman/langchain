
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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3}
)

llm = ChatOpenAI(model="gpt-4o")

# contextualize q prompt
# This system prompt helps the ai understand that it should reformulate the question
# based on the chat history to make it a standalone question

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "Which might reference context in the chat history,"
    "formulate a standalone question which can be understood "
    "without the chat history. Do not ansqer the question, just"
    "reformulate it if needed and otherwise return as it is."
)

# create a prompt template for contextualize question
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# create history aware retri
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# answer question prompt
qa_system_prompt = (
    "You are an assistant for question answering task, Use"
    "the following pieces of retruevr cintext to answer the question"
    "Question: If yoy don't know the answer, you can say 'I don't know'"
    "Use 3 sentance max and keeo0o the answer short and simple"
    "\n\n"
    "{context}"
)

# create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


# create a chain to combine for question and answer
# create_stuff_documents_chain feeds all retrieved context into the llm 
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)  

# create a retrieval chain that combines the history aware retriever and question answer chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def continual_chat():
    print("Start chat with ai")
    chat_history = []
    while True:
        query= input("You: ")
        if query.lower() == "exit":
            break
        # process the users query through the RAG chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})   
        print("AI: ", result['answer'])
        # update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result['answer']))

if __name__ == "__main__":
    continual_chat()


