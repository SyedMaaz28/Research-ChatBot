import asyncio
import os
import streamlit as st

# Fix: Add event loop for grpc.aio inside Streamlit
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("🔬 Research ChatBot using Gemini + LangChain")
st.sidebar.title("URL Loader")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Enter URL {i+1}")
    if url:
        urls.append(url)

process_button = st.sidebar.button("🔄 Process URLs")
FAISS_FOLDER = "faiss_index"
placeholder = st.empty()

# Set up LLM and Embeddings with explicit API key
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro"
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Step 1: Process URLs and Save Vectorstore
if process_button and urls:
    try:
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        placeholder.text("📄 Documents loaded...")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        placeholder.text("✂️ Documents split...")
        
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(FAISS_FOLDER)
        placeholder.success("✅ Documents processed and stored!")
        
    except Exception as e:
        st.error(f"❌ Error while processing: {e}")

# Step 2: Ask Query
query = st.text_input("💬 Ask your question:")

if query:
    if os.path.exists(FAISS_FOLDER):
        try:
            vectorstore = FAISS.load_local(
                FAISS_FOLDER, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=vectorstore.as_retriever()
            )
            result = qa_chain.run(query)
            
            st.subheader("📌 Answer")
            st.write(result)


        except Exception as e:
            st.error(f"❌ Retrieval error: {e}")
    else:
        st.warning("⚠️ Please process some URLs first.")
