import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
persist_directory = str(BASE_DIR / "db" / "chroma_db")

# ----------------------------
# Load embeddings + DB
# ----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

# ----------------------------
# Retriever
# ----------------------------
# top-3 × 400-tok chunks ≈ 1200 tokens — fits comfortably, no truncation needed.
retriever = db.as_retriever(search_kwargs={"k": 3})

# ----------------------------
# IMPROVED: Instruction-tuned LLM
# ----------------------------
# TinyLlama is small (1.1B), fast on CPU, and trained for chat/instruction
# following — it will actually answer the question instead of continuing the prompt.
# Alternatives if you want better accuracy and have more RAM:
#   "microsoft/phi-2"            (~2.7B, better reasoning, slower)
#   "google/gemma-2b-it"         (~2B, very good instruction following)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"⏳ Loading model: {model_name} ...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,   # more room for a complete answer
    do_sample=False,
    temperature=1.0,      # required when do_sample=False
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto"     # uses GPU if available, falls back to CPU
)

print("✅ Model loaded\n")

# ----------------------------
# IMPROVED: Chat-template prompt builder
# ----------------------------
# TinyLlama (and most modern instruction models) are trained with a specific
# chat template. Skipping it causes the model to treat the prompt as raw
# completion text, leading to wandering or irrelevant answers.
def build_prompt(context: str, query: str) -> str:
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# ----------------------------
# Query loop
# ----------------------------
print("Ask your questions (type 'exit' to quit):\n")

while True:
    query = input("You: ").strip()

    if not query:
        continue

    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Step 1: Retrieve
    docs = retriever.invoke(query)

    if not docs:
        print("\n⚠️  No relevant documents found.\n")
        continue

    print("\n--- Retrieved Context ---")
    context_parts = []
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i} ({doc.metadata.get('filename', 'unknown')}):\n{doc.page_content}")
        context_parts.append(doc.page_content)

    # Step 2: Build context — no truncation needed with 400-tok chunks
    context = "\n\n".join(context_parts)

    # Step 3: Build prompt using the model's chat template
    prompt = build_prompt(context, query)

    # Step 4: Generate
    result = pipe(prompt)
    full_text = result[0]["generated_text"] if isinstance(result, list) else result

    # Step 5: Strip the prompt prefix to get only the new answer
    answer = full_text[len(prompt):].strip()

    print("\n--- Final Answer ---")
    print(answer)
    print("\n" + "=" * 50 + "\n")


# Synthetic test questions:
# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"