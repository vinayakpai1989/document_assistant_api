import pdfplumber
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from app.db.memory_store import db
from dotenv import load_dotenv
import os

load_dotenv(override=True)
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASEURL =  os.getenv('DEEPSEEK_BASE_URL')


def process_document(file) -> str:
    """Extract text from PDF, split into chunks, and save in vector DB."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    # Create embeddings + vector DB
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",   # ✅ stable Gemini embedding model
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectordb = Chroma.from_documents(docs, embeddings)

    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    db[doc_id] = vectordb

    llm = ChatOpenAI(
        api_key=deepseek_api_key,
        base_url=DEEPSEEK_BASEURL,
        model="deepseek-chat",
        temperature=0,
    )

    summary_prompt = f"Summarize the following document in a concise way:\n\n{text[:6000]}"

    summary = llm.predict(summary_prompt)

    return doc_id,summary


def answer_question(doc_id: str, question: str, chat_history: list = []):
    """Retrieve relevant info and answer the question."""
    vectordb = db.get(doc_id)
    if not vectordb:
        return "Document not found."

    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(
        api_key=deepseek_api_key,
        base_url=DEEPSEEK_BASEURL,
        model="deepseek-chat",
        temperature=0,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = qa_chain({"question": question, "chat_history": chat_history})

    return result["answer"]
