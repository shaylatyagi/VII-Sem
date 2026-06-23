"""
demo.py
-------
Semantic Cache — Live Demo Script
VII Semester B.Tech CSE Project | VIT Vellore
Team: Shayla & Lohith

Demonstrates:
  1. Query classification with adaptive thresholds
  2. Cache miss → LLM call → saved to cache
  3. Cache hit → instant return
  4. Response time comparison with bar chart
  5. Adaptive threshold vs fixed threshold decision table

Run from the project root:
    python notebooks/demo.py
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from src.classifier import classify_query
from src.cache import check_cache, save_to_cache

print("=" * 70)
print("  SEMANTIC CACHE — DEMO")
print("  VII Sem B.Tech CSE | VIT Vellore | Shayla & Lohith")
print("=" * 70)

# ── 1. CLASSIFIER ─────────────────────────────────────────────────────────────
print("\n── 1. QUERY CLASSIFICATION ──────────────────────────────────────────")
print(f"{'Query':<50} {'Category':<12} {'Threshold'}")
print("-" * 75)
for q in [
    "What is machine learning?",
    "Who invented the telephone?",
    "How does a neural network learn?",
    "Why does the sky appear blue?",
    "Compare supervised and unsupervised learning",
    "Write a short poem about artificial intelligence",
    "Suggest five project ideas for machine learning",
]:
    cat, thr = classify_query(q)
    print(f"{q:<50} {cat:<12} {thr}")

# ── 2. CACHE MISS ─────────────────────────────────────────────────────────────
print("\n── 2. CACHE MISS — LLM CALL ─────────────────────────────────────────")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "openai/gpt-oss-20b"

def ask(query):
    start = time.time()
    cat, thr = classify_query(query)
    cached, sim, _ = check_cache(query)
    if cached:
        ms = round((time.time() - start) * 1000)
        return {"answer": cached, "cache_hit": True, "category": cat,
                "threshold": thr, "similarity": sim, "ms": ms}
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": "Answer clearly and concisely."},
                  {"role": "user", "content": query}]
    )
    answer = resp.choices[0].message.content
    save_to_cache(query, answer)
    ms = round((time.time() - start) * 1000)
    return {"answer": answer, "cache_hit": False, "category": cat,
            "threshold": thr, "similarity": sim, "ms": ms}

q1 = "What is deep learning?"
r1 = ask(q1)
print(f"Query      : {q1}")
print(f"Cache Hit  : {r1['cache_hit']}   Category: {r1['category']}   Time: {r1['ms']} ms")
print(f"Answer     : {r1['answer'][:120]}...")

# ── 3. CACHE HIT ──────────────────────────────────────────────────────────────
print("\n── 3. CACHE HIT — SAME QUERY ────────────────────────────────────────")
r2 = ask(q1)
print(f"Query      : {q1}")
print(f"Cache Hit  : {r2['cache_hit']}    Similarity: {r2['similarity']}   Time: {r2['ms']} ms")
print(f"Speedup    : {r1['ms'] / max(r2['ms'], 1):.1f}x faster")

# ── 4. RESPONSE TIME COMPARISON ───────────────────────────────────────────────
print("\n── 4. RESPONSE TIME COMPARISON ──────────────────────────────────────")
demo_queries = [
    "What is a neural network?",
    "Define gradient descent",
    "What are transformers in NLP?",
]
miss_times, hit_times, labels = [], [], []
for q in demo_queries:
    rm = ask(q)
    rh = ask(q)
    miss_times.append(rm['ms'])
    hit_times.append(rh['ms'])
    labels.append(q[:28] + "..." if len(q) > 28 else q)
    print(f"  Miss: {rm['ms']:>5} ms  |  Hit: {rh['ms']:>4} ms  |  {q}")

print(f"\n  Average miss : {sum(miss_times)/len(miss_times):.0f} ms")
print(f"  Average hit  : {sum(hit_times)/len(hit_times):.0f} ms")
print(f"  Average speedup: {sum(miss_times)/sum(hit_times):.1f}x")

# Bar chart (saves to file)
try:
    import matplotlib.pyplot as plt
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar([i - w/2 for i in x], miss_times, w, label="Cache Miss (LLM call)", color="#e74c3c")
    b2 = ax.bar([i + w/2 for i in x], hit_times,  w, label="Cache Hit (instant)",   color="#2ecc71")
    ax.set_xlabel("Query"); ax.set_ylabel("Response Time (ms)")
    ax.set_title("Semantic Cache: Response Time — Cache Miss vs Cache Hit")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.bar_label(b1, fmt="%d ms", padding=3)
    ax.bar_label(b2, fmt="%d ms", padding=3)
    plt.tight_layout()
    chart_path = os.path.join("notebooks", "response_time_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.show()
    print(f"\n  Chart saved → {chart_path}")
except Exception as e:
    print(f"\n  (Chart skipped: {e})")

# ── 5. ADAPTIVE THRESHOLD DEMO ────────────────────────────────────────────────
print("\n── 5. ADAPTIVE THRESHOLD — WHY FIXED THRESHOLD FAILS ───────────────")
seed_q = "What is photosynthesis?"
seed_a = "Photosynthesis is the process by which plants convert sunlight, CO2, and water into glucose and oxygen."
save_to_cache(seed_q, seed_a)
print(f"Cached: '{seed_q}'\n")

print(f"{'Query':<45} {'Cat':<12} {'Thr':<6} {'Sim':<8} {'Decision'}")
print("-" * 85)
for q in [
    "What is photosynthesis?",
    "Define photosynthesis",
    "Can you explain photosynthesis simply?",
    "Write a poem about photosynthesis",
]:
    cat, thr = classify_query(q)
    cached, sim, _ = check_cache(q)
    decision = "✅ HIT " if cached else "❌ MISS"
    print(f"{q:<45} {cat:<12} {thr:<6} {sim:<8} {decision}")

print("\n  Key insight: 'explain photosynthesis simply' (analytical, thr=0.88)")
print("  has similarity ~0.87 with the cached factual answer → MISS (correct).")
print("  A fixed threshold of 0.80 would return the wrong cached answer here.")

print("\n" + "=" * 70)
print("  DEMO COMPLETE")
print("=" * 70)
