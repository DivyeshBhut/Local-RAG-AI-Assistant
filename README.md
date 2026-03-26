
🤖 Local RAG AI Assistant
==================================================

A local AI-powered Retrieval-Augmented Generation (RAG) system
that allows you to query your own documents efficiently
without relying on external APIs.

--------------------------------------------------
🚀 FEATURES
--------------------------------------------------
- Ask questions from your own documents
- Runs completely locally (no OpenAI API required)
- Fast and lightweight
- Simple and interactive UI

--------------------------------------------------
🏗️ TECH STACK
--------------------------------------------------
- LangChain
- HuggingFace Transformers (TinyLlama)
- Chroma Vector Database
- Streamlit

--------------------------------------------------
📦 SETUP INSTRUCTIONS
--------------------------------------------------

1) Clone the Repository

- git clone Repo

--------------------------------------------------

2) Create Virtual Environment

- python -m venv venv

- Activate Environment:

- Windows : venv\Scripts\activate

- Mac/Linux : source venv/bin/activate

--------------------------------------------------

3) Install Dependencies

- pip install -r requirements.txt

--------------------------------------------------

4) Add Your Documents

- Add your .txt files:
- docs/sample.txt

--------------------------------------------------

5) Run Ingestion Pipeline

- python ingestion_pipeline.py

- This step creates the vector database.

--------------------------------------------------

6) Run the Application

- streamlit run app.py

--------------------------------------------------
💬 USAGE
--------------------------------------------------
- Open browser (it will open automatically)
- Enter your question
- Get answers from your documents

--------------------------------------------------
⚠️ NOTES
--------------------------------------------------
- First run may download models (may take time)
- Works best with .txt files
- Requires Python 3.9+

--------------------------------------------------
👨‍💻 AUTHOR
--------------------------------------------------
- Divyesh Bhut 
- Note : This project is intended for learning RAG concepts.
