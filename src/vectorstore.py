# src/vectorstore.py
import os
import time
import json
from tqdm import tqdm
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from src.helper import load_json, save_json

# ========== 1. Create or Connect to Pinecone Index ==========

def initialize_pinecone_index(index_name, dimension=768, api_key=None, region="us-east-1", recreate=False):
    """Create or connect to a Pinecone index with given settings."""
    if api_key is None:
        raise ValueError("Pinecone API key is required.")

    pc = Pinecone(api_key=api_key)

    existing = [index.name for index in pc.list_indexes().indexes]
    if recreate and index_name in existing:
        pc.delete_index(index_name)
        print(f"🧹 Deleted existing index: {index_name}")

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region)
        )
        print(f"✅ Created new index: {index_name}")
    else:
        print(f"📎 Using existing index: {index_name}")

    return pc.Index(index_name)

# ========== 2. Upload chunks in batches ==========

def upload_chunks_to_pinecone(chunks, embeddings, index_name, batch_size=100, log_file="upload_log.json"):
    """Upload text chunks to Pinecone with batch logging."""
    index = PineconeVectorStore.get_pinecone_index(index_name=index_name)
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    completed_batches = set(load_json(log_file))

    total_batches = len(texts) // batch_size + 1

    for batch_num in tqdm(range(total_batches), desc="📤 Uploading to Pinecone"):
        if batch_num in completed_batches:
            continue

        start = batch_num * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_metas = metadatas[start:end]

        try:
            PineconeVectorStore.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_metas,
                index_name=index_name,
            )
            completed_batches.add(batch_num)
            save_json(log_file, list(completed_batches))

        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            time.sleep(5)
