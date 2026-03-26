import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


# =========================
# Load Documents
# =========================
def load_documents(docs_path):
    print(f"📂 Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} folder not found")

    documents = []

    for file in Path(docs_path).glob("**/*.txt"):
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            if not text:
                print(f"⚠️  Skipping empty file: {file.name}")
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(file), "filename": file.name}
                )
            )

        except Exception as e:
            print(f"⚠️  Skipping {file}: {e}")

    if not documents:
        raise Exception("❌ No documents found. Make sure .txt files exist in the docs/ folder.")

    print(f"✅ Loaded {len(documents)} documents")
    return documents


# =========================
# Split Documents
# =========================
def split_documents(documents):
    print("✂️  Splitting documents...")

    # IMPROVED: Smaller chunks (400 vs 800) so retrieved context
    # fits within model limits without truncation.
    # With top-3 retrieval this gives ~1200 tokens — well within budget.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# =========================
# Create Vector Store
# =========================
def create_vector_store(chunks, persist_directory):
    print("🧠 Creating vector database...")

    # all-MiniLM-L6-v2 is a solid choice — keeping it.
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(persist_directory)
    )

    print(f"✅ Vector DB created — {vectorstore._collection.count()} chunks indexed")
    return vectorstore


# =========================
# Main
# =========================
def main():
    print("\n🚀 === RAG Ingestion Pipeline ===\n")

    docs_path = BASE_DIR / "docs"
    persist_directory = BASE_DIR / "db" / "chroma_db"

    if persist_directory.exists():
        print("🗑️  Deleting old DB...")
        shutil.rmtree(persist_directory)

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    create_vector_store(chunks, persist_directory)

    print("\n🎉 Ingestion completed successfully!\n")


if __name__ == "__main__":
    main()