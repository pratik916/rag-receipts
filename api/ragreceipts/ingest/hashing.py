"""Content hashes for manifest index_hashes — receipts must be traceable to exact
corpus state. Vectors are hashed as little-endian float64 in chunk order; file hashes
cover name + bytes, sorted by path for determinism."""

import hashlib
import struct
from pathlib import Path


def hash_vectors(vectors: list[list[float]]) -> str:
    digest = hashlib.sha256()
    for vector in vectors:
        for value in vector:
            digest.update(struct.pack("<d", value))
    return f"sha256:{digest.hexdigest()}"


def hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"
