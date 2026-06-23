# Semantic Cache for LLM Queries
VII Semester B.Tech CSE Project — VIT Vellore  
Team: Shayla & Lohith | Timeline: July 8 – October 28, 2026

## What This Project Does

Every time you ask an AI a question, it does heavy computation from scratch — even if the same question was asked before. This wastes time and money.

We built a Semantic Cache — a smart layer that sits between the user and the AI. When a question comes in, the system checks if something similar was asked before. If yes (cache hit), it returns the saved answer instantly without calling the LLM. If no (cache miss), it calls the LLM, saves the answer, and returns it.

## The Novel Part — Adaptive Thresholds

Most existing tools like GPTCache use one fixed similarity cutoff for all questions. We showed this causes problems:

| Query Type | Example | Threshold | Why |
|---|---|---|---|
| Factual | "What is DNA?" | 0.95 (strict) | Facts are precise — a 90% similar question may be asking something different |
| Analytical | "How does DNA replication work?" | 0.88 (moderate) | Explanations tolerate some variation |
| Creative | "Write a poem about DNA" | 0.80 (lenient) | Creative prompts have many valid outputs |

A fixed threshold of 0.85 would incorrectly serve a factual definition to someone asking for an explanation. Our system classifies the query first, then applies the right threshold. In our experiments this gives 0 false positives (wrong answers to users) compared to 1–2 false positives with any fixed threshold.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Embeddings | sentence-transformers — all-MiniLM-L6-v2 |
| Vector Database | ChromaDB (persistent, cosine similarity) |
| LLM | Groq API — openai/gpt-oss-20b |
| API | FastAPI + uvicorn |
| Config | python-dotenv |

## Project Structure

```
VII_Sem_Project/
├── src/
│   ├── embedder.py             # Text → embedding vector
│   ├── classifier.py           # Query type → adaptive threshold
│   ├── cache.py                # ChromaDB cache (check + save)
│   └── main.py                 # FastAPI server (/ask, /health)
├── tests/
│   └── test_cache.py           # 4 pytest tests (all passing)
├── experiments/
│   └── compare_thresholds.py   # Adaptive vs fixed threshold comparison
├── notebooks/
│   └── demo.py                 # Demo script with response time comparison
├── .gitignore
└── requirements.txt
```

## Setting Up (Lohith, this is for you)

**Step 1 — Clone the repo**
```powershell
git clone https://github.com/shaylatyagi/VII-Sem.git
cd VII-Sem
```

**Step 2 — Create and activate a virtual environment**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Step 3 — Install dependencies**
```powershell
pip install -r requirements.txt
```
The first time this runs, it will also download the sentence-transformer model (~90MB). That's expected, just let it finish.

**Step 4 — Add your Groq API key**

Create a file called `.env` in the project root (same folder as requirements.txt) and paste this inside:
```
GROQ_API_KEY=your_key_here
```
Get a free key at https://console.groq.com/keys — takes 30 seconds.

**Step 5 — Run the server**
```powershell
uvicorn src.main:app --reload
```
Open http://127.0.0.1:8000/docs in your browser. You'll see the Swagger UI where you can send test queries and see the responses.

**Step 6 — Run the demo script**
```powershell
python notebooks/demo.py
```
This sends a few sample queries, shows cache miss vs cache hit times, and prints the adaptive vs fixed threshold comparison table.

**Step 7 — Run the tests**
```powershell
pytest tests/test_cache.py -v
```
Should show 4 passed.

**Step 8 — Run the threshold comparison experiment**
```powershell
python experiments/compare_thresholds.py
```
This is the main experiment — shows exactly why adaptive thresholds beat fixed ones across 6 test cases.

Note: the `chroma_db/` and `models_cache/` folders are not in the repo — they get created automatically the first time you run anything. Don't worry if you don't see them after cloning.

## Results

From running demo.py:

| Metric | Value |
|---|---|
| Cache hit response time | 18–38 ms |
| Cache miss response time | 700–1100 ms |
| Average speedup on cache hit | 31x |
| False positives with fixed threshold (0.80) | 2 |
| False positives with adaptive threshold | 0 |
