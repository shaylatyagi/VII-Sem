"""
cache.py
--------
The semantic cache layer. Sits between the user's query and the LLM API.

HOW IT WORKS:
  1. User sends a query.
  2. We convert the query to an embedding vector (numerical representation).
  3. We search ChromaDB for the most similar previously-seen query vector.
  4. We compute the similarity score: similarity = 1 - cosine_distance.
     (ChromaDB returns distances; we convert to similarity for intuition.)
  5. We compare the similarity against the adaptive threshold for this
     query's category (factual / analytical / creative).
  6. If similarity >= threshold → CACHE HIT: return the stored answer.
  7. If similarity < threshold  → CACHE MISS: caller must query the LLM.

PERSISTENT STORAGE:
  We use ChromaDB's PersistentClient so the cache survives server restarts.
  Data is stored on disk in the ./chroma_db folder at the project root.
  An in-memory client would lose all cached data every time the server stops.
"""

import hashlib

import chromadb

from src.embedder import get_embedding
from src.classifier import classify_query

# -------------------------------------------------------------------------
# ChromaDB setup — persistent client so cache survives server restarts.
# The path "./chroma_db" is relative to wherever the server is launched from
# (the project root when using `uvicorn src.main:app --reload`).
# -------------------------------------------------------------------------
_chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create the collection that stores our cached query-answer pairs.
# A "collection" in ChromaDB is analogous to a table in a relational database.
# get_or_create_collection is idempotent — safe to call on every startup.
_collection = _chroma_client.get_or_create_collection(
    name="query_cache",
    # cosine distance is standard for sentence embeddings.
    # ChromaDB stores distances (0 = identical, 2 = opposite).
    # We convert to similarity: similarity = 1 - distance.
    metadata={"hnsw:space": "cosine"},
)


def _query_hash(text: str) -> str:
    """
    Generate a stable, unique ID for a query string using MD5 hashing.

    We need a short, deterministic identifier to use as the ChromaDB
    document ID. Using the raw query text as an ID would break on special
    characters and long strings. MD5 gives a fixed-length hex string.

    Args:
        text (str): The query string to hash.

    Returns:
        str: 32-character hexadecimal MD5 hash of the input.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def check_cache(query: str) -> tuple[str | None, float, str]:
    """
    Look up the cache for a semantically similar previous query.

    Steps:
      1. Embed the incoming query.
      2. Classify it to determine the similarity threshold.
      3. Search ChromaDB for the single nearest neighbor.
      4. Convert the returned cosine distance to a similarity score.
      5. If similarity meets the threshold, return the cached answer.

    Args:
        query (str): The user's input query.

    Returns:
        tuple[str | None, float, str]:
            - cached_answer: The stored answer string, or None on miss.
            - similarity_score: Float in [0, 1] — how similar the best
              match was (0 = unrelated, 1 = identical). 0.0 on miss.
            - category: The classified query type (e.g., "factual").
    """
    # Step 1: Convert query to embedding vector
    query_embedding = get_embedding(query)

    # Step 2: Classify query → get the adaptive threshold for this type
    category, threshold = classify_query(query)

    # Step 3: Search ChromaDB for the closest stored embedding.
    # n_results=1 returns only the single best match.
    # We wrap in a try/except in case the collection is empty (first query).
    try:
        results = _collection.query(
            query_embeddings=[query_embedding],  # list of lists
            n_results=1,
            include=["documents", "distances"],  # we need the stored answer and the distance
        )
    except Exception:
        # Collection is empty or query failed — treat as cache miss
        return (None, 0.0, category)

    # Step 4: Extract the result data
    distances = results.get("distances", [[]])[0]   # list of distances for our single query
    documents = results.get("documents", [[]])[0]   # list of stored answers

    # If nothing was returned (empty collection), it's a cache miss
    if not distances or not documents:
        return (None, 0.0, category)

    # ChromaDB cosine distance is in range [0, 2].
    # Convert to similarity: similarity = 1 - distance.
    # similarity = 1.0 means perfect match; 0.0 means completely unrelated.
    best_distance = distances[0]
    similarity_score = round(1.0 - best_distance, 4)

    # Step 5: Compare similarity against the adaptive threshold
    if similarity_score >= threshold:
        # CACHE HIT — return the stored answer along with metadata
        cached_answer = documents[0]
        return (cached_answer, similarity_score, category)

    # CACHE MISS — similarity was too low for this query type
    return (None, similarity_score, category)


def save_to_cache(query: str, answer: str) -> None:
    """
    Store a new query-answer pair in the ChromaDB cache.

    We store:
      - The embedding of the query (used for future similarity searches).
      - The answer text (returned on a future cache hit).
      - The query text itself as the document content.
      - A unique ID derived from hashing the query.

    Args:
        query (str): The original user query.
        answer (str): The LLM-generated answer to store.

    Returns:
        None
    """
    # Compute the embedding vector for this query
    query_embedding = get_embedding(query)

    # Create a unique, stable ID for this query using its MD5 hash.
    # Using the hash means re-saving the same query will upsert (overwrite)
    # the old entry rather than create a duplicate.
    doc_id = _query_hash(query)

    # upsert = insert if new, update if ID already exists.
    # This prevents duplicates if the same query is asked twice.
    _collection.upsert(
        ids=[doc_id],                        # unique identifier for this entry
        embeddings=[query_embedding],         # vector used for similarity search
        documents=[answer],                   # the answer to return on cache hit
        metadatas=[{"query": query}],         # store original query text for debugging
    )
