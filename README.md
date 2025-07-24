# 🔬 Research ChatBot using Gemini + LangChain

A powerful Research Assistant chatbot built with **Google Gemini (Pro & Embeddings)**, **LangChain**, **FAISS**, and **Streamlit**. This tool extracts, processes, and answers queries based on information scraped from any public URLs.

![Screenshot](https://github.com/SyedMaaz28/Research-ChatBot/blob/main/Chatbot.png)

---

## 🚀 Features

- 🌐 Load content from up to 3 URLs
- ✂️ Automatically split documents into chunks
- 🤖 Use Google's Gemini-Pro LLM for answering queries
- 🔍 Search through custom documents using FAISS
- 🧠 Embedding powered by Gemini Embeddings
- 💬 Ask research questions interactively via Streamlit UI

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Python](https://www.python.org/)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/research-chatbot-gemini.git
   cd research-chatbot-gemini

2. **Create a virtual environment**

   ```python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt

4. **Create a .env file**

   ```bash
   GOOGLE_API_KEY=your_gemini_api_key

5. **Run the app**
   ```bash
   streamlit run main.py
