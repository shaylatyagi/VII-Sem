"""
main.py
-------
FastAPI application entry point for the Semantic Cache project.

This file wires together all the components:
  - Loads the Groq API key from the .env file (never hardcoded).
  - Exposes a POST /ask endpoint that:
      1. Checks the semantic cache first.
      2. Returns the cached answer instantly if it's a cache hit.
      3. Calls the Groq LLM on a cache miss, then stores the new answer.
  - Exposes a GET /health endpoint for server liveness checks.

To run the server from the project root:
    uvicorn src.main:app --reload

The --reload flag means the server restarts automatically when you edit code.
"""

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel

from src.cache import check_cache, save_to_cache
from src.classifier import classify_query

# -------------------------------------------------------------------------
# Load environment variables from the .env file.
# This MUST happen before we try to read GROQ_API_KEY.
# load_dotenv() looks for a .env file in the current working directory
# (which is the project root when you run uvicorn from there).
# -------------------------------------------------------------------------
load_dotenv()


# -------------------------------------------------------------------------
# FastAPI app instance.
# The title and description appear in the auto-generated Swagger UI at
# http://127.0.0.1:8000/docs — useful for demoing the project.
# -------------------------------------------------------------------------
app = FastAPI(
    title="Semantic Cache API",
    description=(
        "A smart caching layer for LLM queries. "
        "Uses semantic similarity with adaptive thresholds to avoid "
        "redundant API calls. Built for VIT Vellore B.Tech CSE VII Sem Project."
    ),
    version="1.0.0",
)


# -------------------------------------------------------------------------
# Groq client — initialized once at startup.
# The API key is read from the environment (set via .env file).
# If the key is missing, we raise an error immediately so the problem
# is obvious at startup rather than silently failing on the first request.
# -------------------------------------------------------------------------
_groq_api_key = os.getenv("GROQ_API_KEY")
if not _groq_api_key:
    raise EnvironmentError(
        "GROQ_API_KEY not found. "
        "Please add it to your .env file: GROQ_API_KEY=your_key_here"
    )

_groq_client = Groq(api_key=_groq_api_key)

# The Groq model to use for answering cache misses.
# llama3-8b-8192 is fast, free-tier accessible, and high quality.
GROQ_MODEL = "openai/gpt-oss-20b"

# -------------------------------------------------------------------------
# Request / Response models (Pydantic).
# Pydantic automatically validates incoming JSON and provides clear error
# messages if a required field is missing or the wrong type.
# -------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for the /ask endpoint."""
    query: str  # The user's question or prompt


class QueryResponse(BaseModel):
    """Response body returned by the /ask endpoint."""
    answer: str             # The answer text (from cache or from LLM)
    cache_hit: bool         # True if the answer came from the cache
    category: str           # Query type: "factual", "analytical", or "creative"
    threshold: float        # The similarity threshold used for this query type
    similarity_score: float # Actual similarity of the best cache match (0.0 if miss with no prior entries)
    response_time_ms: int   # Total time taken to produce this response, in milliseconds


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Liveness check endpoint.

    Returns a simple JSON object to confirm the server is running.
    Useful for load balancers, monitoring tools, or just verifying
    the server started correctly.

    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    Main query endpoint. Implements the semantic cache lookup flow.

    Flow:
      1. Record start time for latency measurement.
      2. Check the semantic cache for a similar previous query.
      3a. CACHE HIT:  Return the stored answer immediately (no LLM call).
      3b. CACHE MISS: Call Groq LLM, save the new answer to cache, return it.

    Args:
        request (QueryRequest): JSON body with a "query" field.

    Returns:
        QueryResponse: Answer plus cache metadata and timing.

    Raises:
        HTTPException 500: If the Groq API call fails.
    """
    # Record the wall-clock time before any processing
    start_time = time.time()

    query = request.query.strip()

    # Classify the query to get its category and threshold.
    # We call this here too (it's also called inside check_cache) so we can
    # include the threshold in the response even on a cache miss.
    category, threshold = classify_query(query)

    # -------------------------------------------------------------------------
    # Step 1: Check the semantic cache
    # check_cache returns (answer_or_None, similarity_score, category)
    # -------------------------------------------------------------------------
    cached_answer, similarity_score, detected_category = check_cache(query)

    if cached_answer is not None:
        # CACHE HIT — return immediately without calling the LLM
        elapsed_ms = int((time.time() - start_time) * 1000)

        return QueryResponse(
            answer=cached_answer,
            cache_hit=True,
            category=detected_category,
            threshold=threshold,
            similarity_score=similarity_score,
            response_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # Step 2: CACHE MISS — call the Groq LLM
    # -------------------------------------------------------------------------
    try:
        # Build the chat completion request.
        # We use a simple system prompt to keep answers focused.
        chat_completion = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Answer the user's question clearly and concisely."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )

        # Extract the answer text from the API response
        llm_answer = chat_completion.choices[0].message.content

    except Exception as exc:
        # If the Groq API call fails, return a 500 error with details.
        # We do NOT cache failed responses.
        raise HTTPException(
            status_code=500,
            detail=f"Groq API call failed: {str(exc)}",
        ) from exc

    # -------------------------------------------------------------------------
    # Step 3: Save the new answer to the cache for future reuse
    # -------------------------------------------------------------------------
    save_to_cache(query, llm_answer)

    # Calculate total response time in milliseconds
    elapsed_ms = int((time.time() - start_time) * 1000)

    return QueryResponse(
        answer=llm_answer,
        cache_hit=False,
        category=detected_category,
        threshold=threshold,
        similarity_score=similarity_score,  # similarity of best near-miss (0.0 if cache was empty)
        response_time_ms=elapsed_ms,
    )
