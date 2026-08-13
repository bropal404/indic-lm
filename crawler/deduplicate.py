import hashlib
import re
from typing import Set

from datasketch import MinHash, MinHashLSH


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 5) -> Set[str]:
    tokens = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class Deduplicator:
    def __init__(self, threshold: float = 0.86, num_perm: int = 128):
        self.exact_hashes: Set[str] = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.next_id = 0

    def seen_exact(self, text: str) -> bool:
        digest = content_hash(text)
        if digest in self.exact_hashes:
            return True
        self.exact_hashes.add(digest)
        return False

    def seen_near(self, text: str) -> bool:
        mh = MinHash(num_perm=self.num_perm)
        grams = shingles(text)
        for gram in grams:
            mh.update(gram.encode("utf-8"))
        if self.lsh.query(mh):
            return True
        self.lsh.insert(str(self.next_id), mh)
        self.next_id += 1
        return False
