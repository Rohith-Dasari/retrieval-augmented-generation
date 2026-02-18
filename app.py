import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from pinecone import Pinecone
from google.genai import types

load_dotenv()

app = FastAPI()

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(name=os.getenv("PINECONE_INDEX"))

class Query(BaseModel):
    question: str

def embed_query(q: str):
    r = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=[q],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return r.embeddings[0].values

@app.post("/chat")
async def chat(q: Query):
    q_vec = embed_query(q.question)

    res = index.query(vector=q_vec, top_k=5, include_metadata=True)

    contexts = [m["metadata"]["text"] for m in res.get("matches", []) if "metadata" in m and "text" in m["metadata"]]
    context_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(contexts)])

    prompt = (
        f"Answer using the following user records:\n{context_str}\n\n"
        f"QUERY: {q.question}\nANSWER:"
    )

    gen_resp = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=250,
        )
    )

    answer = gen_resp.text
    return {"answer": answer}
