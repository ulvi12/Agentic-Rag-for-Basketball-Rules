from typing import Optional
from src.ingestion import get_collection
from src.embeddings import embed_query

ALL_LEAGUES = ["NBA", "WNBA", "NCAA", "FIBA"]

def retrieve_per_league(
    query: str,
    n_per_league: int = 35,
    leagues: Optional[list[str]] = None,
) -> list[dict]:

    target_leagues = leagues or ALL_LEAGUES
    collection = get_collection()
    query_embedding = embed_query(query)

    all_chunks = []
    for league in target_leagues:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_per_league,
            where={"league": league},
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            all_chunks.append({"text": doc, "metadata": meta, "distance": round(dist, 4)})

    return all_chunks