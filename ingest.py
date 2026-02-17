import os
import pandas as pd
import pinecone
from dotenv import load_dotenv

from google import genai 

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index_name = os.getenv("PINECONE_INDEX")

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=3072)
index = pinecone.Index(index_name)

df = pd.read_csv("users.csv").fillna("")

def get_embedding(text):
    r = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=[text]
    )
    return r.embeddings[0]

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