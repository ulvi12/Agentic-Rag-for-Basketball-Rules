import os
import hashlib
import chromadb
import time
from cohere.errors import TooManyRequestsError
from dotenv import load_dotenv
from src.embeddings import CohereDocumentEmbeddingFunction
from src.parsers import (
    load_nba_documents,
    load_wnba_documents,
    load_ncaa_documents,
    load_fiba_documents,
)

load_dotenv()

COLLECTION_NAME = "basketball_rules"

LEAGUE_LOADERS = {
    "NBA": load_nba_documents,
    "WNBA": load_wnba_documents,
    "NCAA": load_ncaa_documents,
    "FIBA": load_fiba_documents,
}

def _make_doc_id(league: str, text: str, idx: int) -> str:

    content_hash = hashlib.md5(text.encode()).hexdigest()[:10]
    return f"{league}_{content_hash}_{idx}"

def get_collection():

    host = os.getenv("CHROMA_HOST", "api.trychroma.com")
    tenant = os.getenv("CHROMA_TENANT", "default_tenant")
    database = os.getenv("CHROMA_DATABASE", "default_database")
    api_key = os.getenv("CHROMA_API_KEY", "")

    headers = {}
    if api_key:
        headers["x-chroma-token"] = api_key

    client = chromadb.HttpClient(
        host=host,
        port=443,
        ssl=True,
        tenant=tenant,
        database=database,
        headers=headers,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=CohereDocumentEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    return collection

def ingest(force: bool = False):

    collection = get_collection()
    print(f"Connected to collection '{collection.name}'.")

    all_ids, all_docs, all_metas = [], [], []

    for league, loader in LEAGUE_LOADERS.items():
        print(f"Parsing PDFs for {league}")
        docs = loader()
        print(f"Parsed {len(docs)} chunks for {league}")
        for i, doc in enumerate(docs):
            doc_id = _make_doc_id(league, doc["page_content"], i)
            all_ids.append(doc_id)
            all_docs.append(doc["page_content"])
            all_metas.append(doc["metadata"])

    batch_size = 96
    total_batches = (len(all_docs) + batch_size - 1) // batch_size

    print(f"Found {len(all_docs)} total chunks.")

    for i, start in enumerate(range(0, len(all_docs), batch_size), 1):
        end = start + batch_size
        batch_ids = all_ids[start:end]

        existing = collection.get(ids=batch_ids, include=[])
        if existing and len(existing["ids"]) == len(batch_ids):
            print(f"Skipping batch {i}/{total_batches} (already in DB)")
            continue

        print(f"Uploading batch {i}/{total_batches} (chunks {start}-{min(end, len(all_docs))})")
        while True:
            try:
                collection.upsert(
                    ids=batch_ids,
                    documents=all_docs[start:end],
                    metadatas=all_metas[start:end],
                )
                break
            except Exception as e:
                if "TooManyRequestsError" in str(type(e)) or "rate limit" in str(e).lower():
                    print("Rate Limit Hit, sleeping for 60 seconds before retrying")
                    time.sleep(60)
                else:
                    raise e

    print(f"Ingestion completed. The database contains {collection.count()} documents.")

if __name__ == "__main__":
    ingest()