"""
classifier.py
-------------
The core novelty of this project: adaptive threshold classification.

WHY DIFFERENT THRESHOLDS?
  A single fixed threshold fails in practice because different query types
  tolerate different levels of semantic variation:

  - FACTUAL queries ("What is DNA?") have exact, unchanging answers.
    A cached answer for "What is DNA?" should NOT be reused for
    "What is RNA?" even though these are similar topics. We need a
    HIGH threshold (0.95) to ensure we only return cached answers
    when the question is almost identical.

  - ANALYTICAL queries ("How does photosynthesis work?") have structured
    explanations that can apply to slightly rephrased questions.
    A MODERATE threshold (0.88) works well here.

  - CREATIVE queries ("Write a poem about rain") can have many valid
    answers. The user just wants creative output, and a cached creative
    response to a similar prompt is likely still useful. A LOW threshold
    (0.80) allows more reuse and saves API calls.

This adaptive approach is what differentiates our project from tools like
GPTCache, which use a single fixed threshold for all query types.
"""


# -------------------------------------------------------------------------
# Keyword lists for each query category.
# We use lowercase matching, so all keywords are lowercase here.
# -------------------------------------------------------------------------

# Factual: questions seeking a specific, objective fact or definition
FACTUAL_KEYWORDS = [
    "what is", "what are", "who is", "who was", "who invented",
    "when did", "when was", "where is", "where was", "define",
    "how many", "how much", "which",
]

# Analytical: questions seeking explanation, reasoning, or comparison
ANALYTICAL_KEYWORDS = [
    "why", "how does", "how do", "compare", "explain", "difference between",
    "analyze", "analyse", "what happens", "what would happen", "pros and cons",
    "advantages", "disadvantages",
]

# Creative: requests to generate, imagine, or produce new content
CREATIVE_KEYWORDS = [
    "write", "create", "suggest", "imagine", "generate", "make",
    "design", "poem", "story", "essay", "draft", "compose", "list some",
]

# Threshold constants — centralised here so changing one value updates
# everything that imports from this module.
THRESHOLD_FACTUAL = 0.95
THRESHOLD_ANALYTICAL = 0.88
THRESHOLD_CREATIVE = 0.80


def classify_query(text: str) -> tuple[str, float]:
    """
    Classify a user query into one of three categories and return the
    appropriate similarity threshold for cache lookup.

    Classification is keyword-based: we check if the lowercase query
    contains any of the known trigger phrases for each category.
    Priority order: factual → creative → analytical → default(analytical).

    Args:
        text (str): The raw user query string.

    Returns:
        tuple[str, float]: A pair of (category_name, threshold).
            - category_name: one of "factual", "analytical", "creative"
            - threshold: the minimum cosine similarity required for a
              cache hit for this type of query.

    Examples:
        >>> classify_query("What is photosynthesis?")
        ('factual', 0.95)

        >>> classify_query("Write a poem about the ocean")
        ('creative', 0.80)

        >>> classify_query("Compare SQL and NoSQL databases")
        ('analytical', 0.88)
    """
    # Normalize to lowercase for case-insensitive keyword matching
    lower_text = text.lower()

    # --- Check FACTUAL first (strictest — highest threshold) ---
    # Factual queries need the closest match because facts are precise.
    for keyword in FACTUAL_KEYWORDS:
        if keyword in lower_text:
            return ("factual", THRESHOLD_FACTUAL)

    # --- Check CREATIVE next (most lenient — lowest threshold) ---
    # Creative prompts can vary widely in wording but still be similar in intent.
    for keyword in CREATIVE_KEYWORDS:
        if keyword in lower_text:
            return ("creative", THRESHOLD_CREATIVE)

    # --- Check ANALYTICAL ---
    for keyword in ANALYTICAL_KEYWORDS:
        if keyword in lower_text:
            return ("analytical", THRESHOLD_ANALYTICAL)

    # --- Default: treat as analytical if no keywords matched ---
    # Analytical is the safest middle-ground default.
    return ("analytical", THRESHOLD_ANALYTICAL)
