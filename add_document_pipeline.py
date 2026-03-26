import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
persist_directory = str(BASE_DIR / "db" / "chroma_db")
docs_path = BASE_DIR / "docs"

# ----------------------------
# Load existing Vector DB
# ----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

# IMPROVED: Use same splitter settings as ingest.py (400/50).
# Keeping them in sync matters — if you re-ingest with different chunk sizes
# the DB becomes inconsistent and retrieval quality drops.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# ----------------------------
# Get already-indexed sources
# ----------------------------
def get_indexed_sources() -> set:
    existing = db.get()
    sources = set()
    for metadata in existing["metadatas"]:
        if metadata and "source" in metadata:
            sources.add(metadata["source"])
    return sources


# ----------------------------
# Add a single document
# ----------------------------
def add_document(file_path: Path, indexed_sources: set) -> int:
    if str(file_path) in indexed_sources:
        print(f"⏭️   Skipping (already indexed): {file_path.name}")
        return 0

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if not text:
            print(f"⚠️   Skipping empty file: {file_path.name}")
            return 0

        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": str(file_path), "filename": file_path.name}]
        )

        db.add_documents(chunks)
        print(f"✅  Added: {file_path.name} ({len(chunks)} chunks)")
        return len(chunks)

    except Exception as e:
        print(f"⚠️   Error processing {file_path.name}: {e}")
        return 0


# ----------------------------
# Scan and sync entire docs/
# ----------------------------
def sync_docs_folder():
    print(f"\n🔍 Scanning: {docs_path}\n")

    if not docs_path.exists():
        raise FileNotFoundError(f"docs/ folder not found at {docs_path}")

    indexed_sources = get_indexed_sources()
    print(f"📦 Already indexed files : {len(indexed_sources)}")

    all_files = list(docs_path.glob("**/*.txt"))
    print(f"📂 Total files in docs/  : {len(all_files)}\n")

    # IMPROVED: Also detect and warn about deleted files still in the DB.
    # Files that exist in the DB but not on disk are stale — log them so
    # the user knows to re-ingest if needed.
    all_file_paths = {str(f) for f in all_files}
    stale = indexed_sources - all_file_paths
    if stale:
        print("⚠️  Stale entries in DB (files deleted from disk):")
        for s in sorted(stale):
            print(f"   - {s}")
        print("   Run ingest.py to rebuild the DB cleanly.\n")

    new_files = 0
    total_chunks = 0

    for file in all_files:
        chunks_added = add_document(file, indexed_sources)
        if chunks_added > 0:
            new_files += 1
            total_chunks += chunks_added

    print(f"\n📊 Summary:")
    print(f"   - New files added  : {new_files}")
    print(f"   - Chunks added     : {total_chunks}")
    print(f"   - Total in DB      : {db._collection.count()}")
    print(f"\n🎉 Sync complete!\n")


if __name__ == "__main__":
    sync_docs_folder()