import streamlit as st
from rag_pipeline import RAGPipeline


st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("🤖 Local RAG Assistant")

# Load pipeline once
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

query = st.text_input("💬 Ask a question:")

if query:
    with st.spinner("Thinking..."):
        answer, docs = pipeline.query(query)

    st.subheader("🤖 Answer")
    st.write(answer)

    # Debug view
    with st.expander("🔍 Retrieved Context"):
        for i, doc in enumerate(docs):
            st.write(f"**Chunk {i+1} ({doc.metadata.get('filename')})**")
            st.write(doc.page_content[:300])
            st.write("---")