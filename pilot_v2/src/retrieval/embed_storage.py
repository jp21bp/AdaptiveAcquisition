"""
Saves and loads dense embeddings to avoid overhead.
In pilot_v2 it won't be needed since there are only 10 questions
"""

from pathlib import Path
import numpy as np

def save_embeddings(
    path: str,
    embeddings: np.ndarray,
):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(path, embeddings)

def load_embeddings(
    path: str,
):
    return np.load(path)