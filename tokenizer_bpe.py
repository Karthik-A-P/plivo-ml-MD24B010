"""tokenizer.py — BPE trained on train_corpus.txt, byte-level fallback
keeps it lossless on arbitrary UTF-8."""
import json, os, re
from collections import Counter

_MERGES_PATH = os.path.join(os.path.dirname(__file__), "bpe_merges.json")
_SPLIT_RE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+")

class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = merges or []
        self.merge_ranks = {tuple(p): i for i, (p, _) in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for i, (pair, new_id) in enumerate(self.merges):
            a, b = pair
            self.id_to_bytes[new_id] = self.id_to_bytes[a] + self.id_to_bytes[b]

    def _merge_word(self, ids):
        ids = list(ids)
        while len(ids) >= 2:
            pairs = [(ids[i], ids[i+1]) for i in range(len(ids)-1)]
            ranked = [(self.merge_ranks[p], i) for i, p in enumerate(pairs) if p in self.merge_ranks]
            if not ranked:
                break
            _, i = min(ranked)
            new_id = 256 + self.merge_ranks[(ids[i], ids[i+1])]
            ids = ids[:i] + [new_id] + ids[i+2:]
        return ids

    def encode(self, text):
        out = []
        for w in _SPLIT_RE.findall(text):
            out.extend(self._merge_word(list(w.encode("utf-8"))))
        return out

    def decode(self, ids):
        return b"".join(self.id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path=_MERGES_PATH):
        json.dump({"merges": [[list(p), nid] for p, nid in self.merges]}, open(path, "w"))

    @classmethod
    def load_from_file(cls, path=_MERGES_PATH):
        data = json.load(open(path))
        return cls(merges=[(tuple(p), nid) for p, nid in data["merges"]])


def train_bpe(text, target_vocab_size):
    word_counts = Counter(_SPLIT_RE.findall(text))
    words = {w: [list(w.encode("utf-8")), c] for w, c in word_counts.items()}
    merges, next_id = [], 256
    n_merges = target_vocab_size - 256
    for m in range(n_merges):
        if m % 100 == 0:
            print(f"merge {m}/{n_merges}")   # ADD THIS
        pair_counts = Counter()
        for ids, freq in words.values():
            for i in range(len(ids) - 1):
                pair_counts[(ids[i], ids[i+1])] += freq
        if not pair_counts:
            break
        best = pair_counts.most_common(1)[0][0]
        merges.append((best, next_id))
        a, b = best
        for w, (ids, freq) in words.items():
            if a in ids and b in ids:
                new_ids, i = [], 0
                while i < len(ids):
                    if i < len(ids)-1 and ids[i] == a and ids[i+1] == b:
                        new_ids.append(next_id); i += 2
                    else:
                        new_ids.append(ids[i]); i += 1
                words[w] = [new_ids, freq]
        next_id += 1
    return BPETokenizer(merges=merges)


def load(path=None):
    if os.path.exists(_MERGES_PATH):
        return BPETokenizer.load_from_file()
    return BPETokenizer(merges=[])