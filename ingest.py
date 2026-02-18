import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from google import genai 
from google.genai import types

load_dotenv()

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1")
        )
    )
index = pc.Index(name=index_name)

df = pd.read_csv("users.csv").fillna("")

def get_embedding(text):
    r = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return r.embeddings[0].values

batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    to_upsert = []
    for j, row in batch.iterrows():
        text = f"{row.name} | {row.email} | {row.city} | {row.job} | {row.bio}"
        emb = get_embedding(text)
        to_upsert.append((str(j), emb, {"text": text}))
    index.upsert(vectors=to_upsert)

print("Ingestion to Pinecone completed")