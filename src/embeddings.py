import os
import cohere
from typing import List
from chromadb import EmbeddingFunction, Embeddings

EMBEDDING_MODEL = "embed-english-v3.0"

_client: cohere.ClientV2 | None = None

def _get_client() -> cohere.ClientV2:
    global _client
    if _client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")
        _client = cohere.ClientV2(api_key=api_key)
    return _client

class CohereDocumentEmbeddingFunction(EmbeddingFunction):

    def __call__(self, input: List[str]) -> Embeddings:
        if not input:
            return []

        client = _get_client()

        response = client.embed(
            texts=input,
            model=EMBEDDING_MODEL,
            input_type="search_document",
            embedding_types=["float"]
        )
        return response.embeddings.float_

def embed_query(query: str) -> list[float]:

    client = _get_client()
    response = client.embed(
        texts=[query],
        model=EMBEDDING_MODEL,
        input_type="search_query",
        embedding_types=["float"]
    )
    return response.embeddings.float_[0]