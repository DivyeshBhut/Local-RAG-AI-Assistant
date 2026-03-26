from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
persist_directory = str(BASE_DIR / "db" / "chroma_db")


class RAGPipeline:
    def __init__(self):
        print("⏳ Loading RAG pipeline...")

        # Embeddings + DB
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )

        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})

        # LLM
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            do_sample=False,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            device_map="auto"
        )

        print("✅ RAG pipeline ready\n")

    # ----------------------------
    # Prompt builder
    # ----------------------------
    def build_prompt(self, context: str, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise question-answering assistant. "
                    "Answer using ONLY the context provided below. "
                    "Be short, factual, and direct. "
                    "If the answer is not in the context, say exactly: 'Not found in context.'"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # ----------------------------
    # Main Query Function
    # ----------------------------
    def query(self, user_query: str):
        # Step 1: Retrieve
        docs = self.retriever.invoke(user_query)

        if not docs:
            return "No relevant documents found.", []

        # Step 2: Build context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 3: Prompt
        prompt = self.build_prompt(context, user_query)

        # Step 4: Generate
        result = self.pipe(prompt)
        full_text = result[0]["generated_text"]

        answer = full_text[len(prompt):].strip()

        return answer, docs