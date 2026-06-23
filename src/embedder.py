"""
embedder.py
-----------
Responsible for converting text (queries or answers) into numerical vectors
called "embeddings". These vectors capture the semantic meaning of the text,
so two sentences that mean the same thing will have vectors that are close
together in space, even if the exact words differ.

Model: all-MiniLM-L6-v2
  - Lightweight (22M parameters), fast, and accurate for sentence similarity.
  - Produces 384-dimensional vectors.
  - Trained specifically for semantic textual similarity tasks.
  - Ideal for our use case: checking if two questions mean the same thing.
"""

import os

# -------------------------------------------------------------------------
# Redirect the HuggingFace model cache to a folder inside the project.
# The default cache (C:\Users\<user>\.cache\huggingface) requires write
# permission to the user profile, which is sometimes denied on shared or
# managed Windows machines. Pointing it at ./models_cache (inside the
# project root) avoids this entirely.
# This MUST be set before importing sentence_transformers so the library
# picks up the correct path at startup.
# -------------------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(_project_root, "models_cache"))

from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------------
# Load the model ONCE at module level.
# Loading inside the function would reload the model on every single call,
# which is slow (~1-2 seconds per call). Module-level loading happens only
# once when the module is first imported, then reuses the same object.
# -------------------------------------------------------------------------
_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list:
    """
    Convert a text string into a semantic embedding vector.

    An embedding is a list of floating-point numbers (a vector) that
    represents the meaning of the text in a high-dimensional space.
    Sentences with similar meanings will have similar (close) vectors.

    Args:
        text (str): The input text to embed (e.g., a user query).

    Returns:
        list: A list of 384 floats representing the semantic embedding.

    Example:
        >>> vec = get_embedding("What is photosynthesis?")
        >>> len(vec)
        384
    """
    # encode() returns a NumPy array; .tolist() converts it to a plain
    # Python list, which is required by ChromaDB for storage.
    embedding = _model.encode(text)
    return embedding.tolist()
