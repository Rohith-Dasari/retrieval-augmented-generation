import os
import pinecone
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

load_dotenv()

app = FastAPI()

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index = pinecone.Index(os.getenv("PINECONE_INDEX"))

class Query(BaseModel):
    question: str

def embed_query(q: str):
    r = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=[q]
    )
    return r.embeddings[0]

@app.post("/chat")
async def chat(q: Query):
    q_vec = embed_query(q.question)

    res = index.query(q_vec, top_k=5, include_metadata=True)

    contexts = [m["metadata"]["text"] for m in res["matches"]]
    context_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(contexts)])

    prompt = (
        f"Answer using the following user records:\n{context_str}\n\n"
        f"QUERY: {q.question}\nANSWER:"
    )

    gen_resp = gemini.models.generate_text(
        model="gemini-pro-1.5",  
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=250
    )

    answer = gen_resp.text
    return {"answer": answer}
