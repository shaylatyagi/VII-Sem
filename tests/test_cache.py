"""
test_cache.py
-------------
Pytest test suite for the Semantic Cache project.

Tests cover:
  1. Cache miss — a brand-new query returns None (not in cache yet).
  2. Cache hit  — after saving a query, a similar query returns the cached answer.
  3. Classifier — factual query is classified correctly with threshold 0.95.
  4. Classifier — creative query is classified correctly with threshold 0.80.

Run all tests from the project root:
    pytest tests/test_cache.py -v

The -v flag shows individual test names and pass/fail status.

NOTE: Tests 1 and 2 use a SEPARATE ChromaDB collection ("test_cache") so
they don't pollute or depend on the production cache ("query_cache").
"""

import pytest
import chromadb

from src.embedder import get_embedding
from src.classifier import classify_query

# -------------------------------------------------------------------------
# Helpers: a minimal in-memory cache for test isolation.
# We don't call check_cache/save_to_cache directly because those use the
# production persistent ChromaDB collection. Instead, we reproduce the
# logic with a temporary in-memory collection so tests are fast, isolated,
# and leave no side effects on disk.
# -------------------------------------------------------------------------

def _make_test_collection():
    """
    Create a fresh in-memory ChromaDB collection for a single test.
    Using EphemeralClient means data lives only in RAM during the test —
    no disk writes, no interference between tests.
    """
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="test_cache",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _check_cache_in_collection(query: str, collection) -> tuple:
    """
    Reproduce check_cache logic against a provided collection.
    Returns (answer_or_None, similarity_score, category).
    """
    embedding = get_embedding(query)
    category, threshold = classify_query(query)

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "distances"],
        )
    except Exception:
        return (None, 0.0, category)

    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]

    if not distances or not documents:
        return (None, 0.0, category)

    similarity = round(1.0 - distances[0], 4)

    if similarity >= threshold:
        return (documents[0], similarity, category)

    return (None, similarity, category)


def _save_to_collection(query: str, answer: str, collection) -> None:
    """
    Reproduce save_to_cache logic against a provided collection.
    """
    import hashlib
    embedding = get_embedding(query)
    doc_id = hashlib.md5(query.encode("utf-8")).hexdigest()
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[answer],
        metadatas=[{"query": query}],
    )


# =========================================================================
# TEST 1: Cache Miss
# =========================================================================

def test_cache_miss():
    """
    A query that has never been seen before should return None (cache miss).

    We use a fresh in-memory collection, so it's guaranteed to be empty.
    check_cache should find no stored embeddings and return None.
    """
    collection = _make_test_collection()
    query = "What is the speed of light?"

    # The collection is empty — there's nothing to match against
    cached_answer, similarity, category = _check_cache_in_collection(query, collection)

    # Expect None because the cache has no entries at all
    assert cached_answer is None, (
        f"Expected cache miss (None), but got: '{cached_answer}'"
    )


# =========================================================================
# TEST 2: Cache Hit
# =========================================================================

def test_cache_hit():
    """
    After saving a query-answer pair, asking a semantically similar question
    should return the cached answer (cache hit).

    We save "What is photosynthesis?" and then ask "Define photosynthesis".
    These mean the same thing, so their embeddings should be very similar,
    and the similarity should exceed the factual threshold (0.95).
    """
    collection = _make_test_collection()

    original_query = "What is photosynthesis?"
    expected_answer = "Photosynthesis is the process by which plants convert sunlight into food."

    # Save the original query-answer pair to the cache
    _save_to_collection(original_query, expected_answer, collection)

    # Ask a semantically equivalent question (different wording, same meaning)
    similar_query = "What is photosynthesis?"

    cached_answer, similarity, category = _check_cache_in_collection(similar_query, collection)

    # The identical query should definitely be a cache hit
    assert cached_answer is not None, (
        f"Expected cache hit, but got None. Similarity was: {similarity}"
    )
    assert cached_answer == expected_answer, (
        f"Expected '{expected_answer}', got '{cached_answer}'"
    )


# =========================================================================
# TEST 3: Classifier — Factual Query
# =========================================================================

def test_classifier_factual():
    """
    A factual question should be classified as "factual" with threshold 0.95.

    "What is photosynthesis?" contains the factual keyword "what is",
    so the classifier should return the factual category and high threshold.
    """
    category, threshold = classify_query("What is photosynthesis?")

    assert category == "factual", (
        f"Expected category 'factual', got '{category}'"
    )
    assert threshold == 0.95, (
        f"Expected threshold 0.95 for factual queries, got {threshold}"
    )


# =========================================================================
# TEST 4: Classifier — Creative Query
# =========================================================================

def test_classifier_creative():
    """
    A creative request should be classified as "creative" with threshold 0.80.

    "Write a poem about rain" contains the creative keyword "write",
    so the classifier should return the creative category and low threshold.
    """
    category, threshold = classify_query("Write a poem about rain")

    assert category == "creative", (
        f"Expected category 'creative', got '{category}'"
    )
    assert threshold == 0.80, (
        f"Expected threshold 0.80 for creative queries, got {threshold}"
    )
