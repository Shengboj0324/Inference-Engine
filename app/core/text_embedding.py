"""Shared, dependency-free deterministic text embedding.

The canonical bag-of-words hashing embedder used as the no-dependency fallback
across the codebase (context memory, the embedding backend, candidate
retrieval).  Extracted here so there is exactly **one** implementation rather
than copies in each consumer.

Determinism matters: token → bucket assignment uses :func:`zlib.crc32` over the
UTF-8 bytes (stable across processes/platforms/Python versions), *not* the
salted built-in :func:`hash`, so a vector persisted in one process still matches
a query embedded in another.
"""

from __future__ import annotations

import math
import zlib
from typing import List

#: Default fixed embedding dimension.
DEFAULT_DIM: int = 512


def stable_token_bucket(token: str, dim: int = DEFAULT_DIM) -> int:
    """Map ``token`` to a stable hashing-trick bucket in ``[0, dim)``."""
    return zlib.crc32(token.encode("utf-8")) % dim


def bow_embed(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    """Return a deterministic, L2-normalised bag-of-words embedding of ``text``."""
    vec: List[float] = [0.0] * dim
    for token in (text or "").lower().split():
        vec[stable_token_bucket(token, dim)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
