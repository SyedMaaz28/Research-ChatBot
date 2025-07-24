import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from typing import Any, List, Optional

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if API key is loaded
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in environment variables!")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Custom Gemini LLM wrapper
class GeminiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            model = genai.GenerativeModel('models/gemini-2.5-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ✅ Streamlit UI
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

# ✅ Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm():
    return GeminiLLM()

# ✅ Step 1: Process URLs and Save Vectorstore
if process_button and urls:
    try:
        embeddings = get_embeddings()
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

# ✅ Step 2: Ask Query
query = st.text_input("💬 Ask your question:")

if query:
    if os.path.exists(FAISS_FOLDER):
        try:
            embeddings = get_embeddings()
            llm = get_llm()

            vectorstore = FAISS.load_local(
                FAISS_FOLDER,
                embeddings,
                allow_dangerous_deserialization=True
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )

            with st.spinner("🤔 Thinking..."):
                result = qa_chain.run(query)

            st.subheader("📌 Answer")
            st.write(result)

        except Exception as e:
            st.error(f"❌ Retrieval error: {e}")
    else:
        st.warning("⚠️ Please process some URLs first.")
