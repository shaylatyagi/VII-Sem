"""
compare_thresholds.py
---------------------
Experiment: Adaptive Threshold vs Fixed Threshold

This script validates the core research contribution of the project:
adaptive per-type thresholds make SAFER caching decisions than any
single fixed threshold.

KEY INSIGHT — two types of mistakes:
  False Positive (FP): Cache returns an answer when it SHOULD call the LLM.
                       → User receives a WRONG or mismatched answer.
                       → This is the SERIOUS error.

  False Negative (FN): Cache calls LLM when the cache COULD have been used.
                       → User receives a correct answer, just slower.
                       → This wastes a little time/money but is NOT harmful.

A fixed threshold cannot avoid both errors simultaneously:
  - Low threshold (e.g. 0.80): catches more cache hits, but risks FPs
    (returning a factual definition when the user asked for an explanation).
  - High threshold (e.g. 0.90): avoids FPs, but misses valid cache hits (FNs).

Our adaptive system is designed to ELIMINATE false positives entirely by
applying a stricter threshold to query types where a mismatch is harmful
(factual), and a more lenient threshold where reuse is safe (creative).

HOW TO RUN:
    python experiments/compare_thresholds.py
    (from the project root: VII_Sem_Project/)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
import chromadb

from src.embedder import get_embedding
from src.classifier import classify_query

# ── Constants ─────────────────────────────────────────────────────────────────

FIXED_THRESHOLDS = [0.80, 0.85, 0.90]

ADAPTIVE = {"factual": 0.95, "analytical": 0.88, "creative": 0.80}

# ── Test Dataset ───────────────────────────────────────────────────────────────
#
# error_type classifies WHY a wrong decision is wrong:
#   "false_positive" — cache returned a mismatched answer (BAD: wrong answer)
#   "false_negative" — cache missed a valid reuse (MILD: just an extra LLM call)
#
TEST_CASES = [
    # ── Identical queries: should always hit ───────────────────────────────────
    {
        "id": 1,
        "seed_query":  "What is photosynthesis?",
        "seed_answer": "Photosynthesis is the process by which plants convert sunlight, CO2, and water into glucose and oxygen.",
        "test_query":  "What is photosynthesis?",
        "correct":     "hit",
        "error_type":  "false_negative",   # if wrong: unnecessary LLM call
        "note":        "Identical factual query — cache must be used.",
    },
    # ── Cross-type: factual cached answer served to analytical request ─────────
    {
        "id": 2,
        "seed_query":  "What is photosynthesis?",
        "seed_answer": "Photosynthesis is the process by which plants convert sunlight, CO2, and water into glucose and oxygen.",
        "test_query":  "Can you explain photosynthesis in simple terms?",
        "correct":     "miss",
        "error_type":  "false_positive",   # if wrong: factual definition served as an explanation
        "note":        "Analytical query — user wants a simple EXPLANATION, not a definition. Cache must NOT be used.",
    },
    # ── Cross-type: factual cached answer served to creative request ───────────
    {
        "id": 3,
        "seed_query":  "What is photosynthesis?",
        "seed_answer": "Photosynthesis converts sunlight into glucose.",
        "test_query":  "Write a short poem about photosynthesis",
        "correct":     "miss",
        "error_type":  "false_positive",   # if wrong: a definition is returned instead of a poem
        "note":        "Creative query — user wants a POEM. A scientific definition is completely wrong.",
    },
    # ── Same concept, different creative phrasing: should hit ─────────────────
    {
        "id": 4,
        "seed_query":  "Suggest ideas for a machine learning project",
        "seed_answer": "1. Image classifier\n2. Sentiment analysis\n3. Recommendation system\n4. Fraud detection",
        "test_query":  "Generate project ideas for machine learning",
        "correct":     "hit",
        "error_type":  "false_negative",   # if wrong: LLM called when cache had a valid answer
        "note":        "Both are creative requests for the same thing — cached suggestions are valid.",
    },
    # ── Analytical: similar analytical question should hit ────────────────────
    {
        "id": 5,
        "seed_query":  "Why is the sky blue?",
        "seed_answer": "The sky appears blue because of Rayleigh scattering — shorter blue wavelengths scatter more than red when sunlight hits the atmosphere.",
        "test_query":  "Why does the sky look blue?",
        "correct":     "hit",
        "error_type":  "false_negative",
        "note":        "Nearly identical analytical query — cached explanation is valid.",
    },
    # ── Analytical: slightly different analytical question ────────────────────
    {
        "id": 6,
        "seed_query":  "What is a neural network?",
        "seed_answer": "A neural network is a computational model inspired by the human brain, consisting of layers of interconnected nodes that process information.",
        "test_query":  "How does a neural network work?",
        "correct":     "miss",
        "error_type":  "false_positive",   # if wrong: a definition is served when user wants a how-it-works explanation
        "note":        "Different intent — 'what is' vs 'how does'. Definition != explanation of mechanism.",
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fresh():
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection("exp", metadata={"hnsw:space": "cosine"})

def _save(q, a, col):
    col.upsert(
        ids=[hashlib.md5(q.encode()).hexdigest()],
        embeddings=[get_embedding(q)],
        documents=[a],
        metadatas=[{"query": q}],
    )

def _check(q, threshold, col):
    r = col.query(query_embeddings=[get_embedding(q)], n_results=1,
                  include=["documents", "distances"])
    distances = r.get("distances", [[]])[0]
    if not distances:
        return ("miss", 0.0)
    sim = round(1.0 - distances[0], 4)
    return ("hit" if sim >= threshold else "miss", sim)

# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 80)
    print("  EXPERIMENT: Adaptive Threshold vs Fixed Threshold")
    print("  VII Semester B.Tech CSE Project — VIT Vellore")
    print("=" * 80)
    print("\n  FALSE POSITIVE = cache returns wrong answer (HARMFUL)")
    print("  FALSE NEGATIVE = cache missed, LLM called instead (HARMLESS)")

    # Track per-system: FP count, FN count
    stats = {f"fixed_{t}": {"fp": 0, "fn": 0} for t in FIXED_THRESHOLDS}
    stats["adaptive"] = {"fp": 0, "fn": 0}

    print(f"\n  Running {len(TEST_CASES)} test cases...\n")

    for case in TEST_CASES:
        sq, sa, tq = case["seed_query"], case["seed_answer"], case["test_query"]
        correct = case["correct"]
        etype   = case["error_type"]

        # Get similarity once
        col0 = _fresh(); _save(sq, sa, col0)
        _, sim = _check(tq, 0.0, col0)

        # Classify for adaptive
        category, adaptive_thr = classify_query(tq)

        print(f"  Case {case['id']}: \"{tq[:60]}\"")
        print(f"    Seed : \"{sq}\"")
        print(f"    Similarity  : {sim}  |  Category: {category}  |  Adaptive threshold: {adaptive_thr}")
        print(f"    Correct call: {correct.upper()}  (if wrong → {etype.replace('_', ' ').upper()})")
        print(f"    {'─'*68}")

        for ft in FIXED_THRESHOLDS:
            col = _fresh(); _save(sq, sa, col)
            decision, _ = _check(tq, ft, col)
            ok = (decision == correct)
            if not ok:
                if etype == "false_positive": stats[f"fixed_{ft}"]["fp"] += 1
                else:                         stats[f"fixed_{ft}"]["fn"] += 1
            marker = "✅" if ok else f"❌ {etype.replace('_',' ').upper()}"
            print(f"    Fixed {ft} → {decision.upper():5}  {marker}")

        col = _fresh(); _save(sq, sa, col)
        decision, _ = _check(tq, adaptive_thr, col)
        ok = (decision == correct)
        if not ok:
            if etype == "false_positive": stats["adaptive"]["fp"] += 1
            else:                         stats["adaptive"]["fn"] += 1
        marker = "✅" if ok else f"❌ {etype.replace('_',' ').upper()}"
        print(f"    Adaptive    → {decision.upper():5}  {marker}  (threshold={adaptive_thr})")
        print()

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(TEST_CASES)
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"\n  {'System':<20} {'False Positives':>16} {'False Negatives':>16}  Verdict")
    print(f"  {'─'*20} {'(wrong answers)':>16} {'(extra LLM calls)':>16}")
    print(f"  {'─'*72}")

    rows = []
    for ft in FIXED_THRESHOLDS:
        k = f"fixed_{ft}"
        rows.append((f"Fixed {ft}", stats[k]["fp"], stats[k]["fn"]))
    rows.append(("Adaptive (ours)", stats["adaptive"]["fp"], stats["adaptive"]["fn"]))

    for name, fp, fn in rows:
        if fp == 0:
            verdict = "✅ No wrong answers"
        else:
            verdict = f"❌ {fp} wrong answer(s) to users"
        print(f"  {name:<20} {fp:>16} {fn:>16}  {verdict}")

    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)

    adaptive_fp = stats["adaptive"]["fp"]
    best_fixed_fp = min(stats[f"fixed_{t}"]["fp"] for t in FIXED_THRESHOLDS)

    print()
    if adaptive_fp == 0 and best_fixed_fp > 0:
        print("  ✅ Adaptive threshold system produces ZERO false positives.")
        print("     No user ever receives a wrong cached answer.")
        print()
        print("     Fixed thresholds make an unavoidable tradeoff:")
        print("       - Low threshold (0.80): more cache hits, but returns wrong")
        print("         answers when query type changes (e.g., definition served")
        print("         in response to a request for explanation or a poem).")
        print("       - High threshold (0.90): fewer wrong answers, but misses")
        print("         many valid cache hits — defeats the purpose of caching.")
        print()
        print("     Our adaptive system resolves this tradeoff by applying")
        print("     strict thresholds where mismatches are harmful (factual)")
        print("     and lenient thresholds where reuse is safe (creative).")
    elif adaptive_fp == best_fixed_fp:
        print(f"  Adaptive and best fixed threshold both achieve {adaptive_fp} false positive(s).")
        print("  Adaptive threshold provides more principled, explainable decisions.")
    else:
        print(f"  Results: Adaptive FP={adaptive_fp}, Best Fixed FP={best_fixed_fp}.")
    print()


if __name__ == "__main__":
    run()
