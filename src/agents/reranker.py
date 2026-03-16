import os
import cohere
from collections import defaultdict

RERANK_MODEL = "rerank-v4.0-fast"
TOP_K_PER_LEAGUE = 20

MIN_RELEVANCE_SCORE = 0.01

_client = None

def _get_client():

    global _client
    if _client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY env variable missing")

        _client = cohere.ClientV2(api_key=api_key)
    return _client

def rerank(query: str, chunks: list[dict], top_k: int = TOP_K_PER_LEAGUE) -> list[dict]:

    if not chunks:
        return []

    client = _get_client()
    docs = [chunk["text"] for chunk in chunks]

    response = client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=len(docs)
    )

    for result in response.results:
        original_idx = result.index
        chunks[original_idx]["rerank_score"] = round(result.relevance_score, 4)

    relevant = [c for c in chunks if c.get("rerank_score", 0) >= MIN_RELEVANCE_SCORE]

    if not relevant:
        return []

    by_league: dict[str, list[dict]] = defaultdict(list)
    for c in relevant:
        by_league[c["metadata"]["league"]].append(c)

    filtered_chunks: list[dict] = []
    for league, league_chunks in by_league.items():
        league_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        filtered_chunks.extend(league_chunks[:top_k])

    filtered_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

    return filtered_chunks