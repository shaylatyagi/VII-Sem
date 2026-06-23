# Semantic Cache for LLM Queries
### VII Semester B.Tech CSE Project — VIT Vellore
**Team:** Shayla & Lohith | **Timeline:** July 8 – October 28, 2026

---

## What This Project Does

Every time you ask an AI a question, it does heavy computation from scratch — even if the same question was asked before. This wastes time and money.

We built a **Semantic Cache** — a smart layer that sits between the user and the AI:

1. A question comes in.
2. The system checks: *"Has someone asked something like this before?"*
3. **Cache hit** → return the saved answer instantly (no AI call needed).
4. **Cache miss** → call the AI, save the answer, return it.

### The Novel Part — Adaptive Thresholds

Most existing tools (like GPTCache) use **one fixed similarity cutoff** for all questions. We showed this is wrong:

| Query Type | Example | Threshold | Why |
|---|---|---|---|
| **Factual** | "What is DNA?" | 0.95 (strict) | Facts are precise — a 90% similar question may be asking something different |
| **Analytical** | "How does DNA replication work?" | 0.88 (moderate) | Explanations tolerate some variation |
| **Creative** | "Write a poem about DNA" | 0.80 (lenient) | Creative prompts have many valid outputs |

A fixed threshold of 0.85 would incorrectly serve a factual definition to someone asking for an explanation. Our system classifies the query first, then applies the right threshold.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Database | `ChromaDB` (persistent, cosine similarity) |
| LLM | Groq API — `openai/gpt-oss-20b` |
| API | `FastAPI` + `uvicorn` |
| Config | `python-dotenv` |

---

## Project Structure

```
VII_Sem_Project/
├── src/
│   ├── embedder.py       # Text → embedding vector
│   ├── classifier.py     # Query type → adaptive threshold
│   ├── cache.py          # ChromaDB cache (check + save)
│   └── main.py           # FastAPI server (/ask, /health)
├── tests/
│   └── test_cache.py     # 4 pytest tests
├── experiments/
│   └── compare_thresholds.py  # Adaptive vs fixed threshold comparison
├── notebooks/
│   └── semantic_cache_demo.ipynb  # Demo with charts
├── chroma_db/            # Auto-created — persistent vector store
├── models_cache/         # Auto-created — HuggingFace model cache
├── .env                  # GROQ_API_KEY (not committed to git)
├── requirements.txt
└── venv/
```

---

## Setup

**1. Activate the virtual environment:**
```powershell
venv\Scripts\activate
```

**2. Install dependencies** (first time only):
```powershell
pip install -r requirements.txt
```

**3. Add your Groq API key to `.env`:**
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com/keys](https://console.groq.com/keys)

---

## Running the Server

From the project root:
```powershell
uvicorn src.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

### Example Request
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is photosynthesis?"}'
```

### Example Response
```json
{
  "answer": "Photosynthesis is the process by which plants convert sunlight...",
  "cache_hit": false,
  "category": "factual",
  "threshold": 0.95,
  "similarity_score": 0.0,
  "response_time_ms": 1020
}
```

---

## Running Tests

```powershell
pytest tests/test_cache.py -v
```

Expected output:
```
tests/test_cache.py::test_cache_miss        PASSED
tests/test_cache.py::test_cache_hit         PASSED
tests/test_cache.py::test_classifier_factual  PASSED
tests/test_cache.py::test_classifier_creative PASSED
4 passed
```

---

## Running the Comparison Experiment

```powershell
python experiments/compare_thresholds.py
```

Shows how adaptive thresholds outperform a fixed threshold by correctly handling borderline cases.

---

## Key Results

| Metric | Value |
|---|---|
| Cache hit response time | ~10–50 ms |
| Cache miss response time | ~1000–1500 ms |
| Speedup on cache hit | **20–100×** |
| Incorrect responses with fixed threshold (0.85) | Present |
| Incorrect responses with adaptive threshold | None |
