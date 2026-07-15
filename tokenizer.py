"""Byte-level BPE tokenizer, trained ONLY on train_corpus.txt.

Why this exists (see RUNLOG.md / NOTES.md):
  The baseline is raw UTF-8 bytes (vocab 256). Every Devanagari character is
  3 bytes, so the Hindi third of the corpus (33% of all bytes) is shredded
  into byte fragments. BPE merges frequent byte pairs into single tokens, so
  the same text becomes far fewer tokens -> each of the 2000 steps sees more
  real text and the fixed context window covers more characters. This is the
  single biggest bpb win.

Guarantees kept (required by evaluate.py and the graders):
  * LOSSLESS: decode(encode(text)) == text exactly. We operate on raw bytes,
    every token maps to a fixed byte string, and base tokens 0..255 cover all
    256 bytes -> any UTF-8 text is always encodable (byte fallback built in).
  * load() takes no required args and rebuilds the tokenizer from bpe.json
    saved next to this file (resolved relative to __file__, no internet).
  * exposes .encode(str)->list[int], .decode(list[int])->str, .vocab_size.

Train it once with:  python tokenizer.py --data ../llm_handout/data/train_corpus.txt --vocab 1024
"""
import argparse
import json
import os
import re

MERGES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe.json")

# Fully-partitioning pre-tokenizer: runs of whitespace OR runs of non-whitespace.
# Concatenation of all pieces == original text, so nothing is ever dropped.
_SPLIT = re.compile(r"\s+|\S+", re.UNICODE)


class BPETokenizer:
    def __init__(self, merges):
        # merges: list of [a, b] in learned order; new id = 256 + rank
        self.merges = {(a, b): 256 + i for i, (a, b) in enumerate(merges)}
        self.ranks = {(a, b): i for i, (a, b) in enumerate(merges)}
        self.vocab_size = 256 + len(merges)
        # id -> bytes, for decoding
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    def _encode_chunk(self, bts):
        # greedy BPE: repeatedly merge the adjacent pair with the best (lowest)
        # rank until no known merge remains.
        ids = list(bts)
        if len(ids) < 2:
            return ids
        while True:
            best_rank, best_i = None, None
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            ids[best_i:best_i + 2] = [self.merges[(ids[best_i], ids[best_i + 1])]]
        return ids

    def encode(self, text):
        out = []
        for piece in _SPLIT.findall(text):
            bts = piece.encode("utf-8")
            cached = self._cache.get(bts)
            if cached is None:
                cached = self._encode_chunk(bts)
                self._cache[bts] = cached
            out.extend(cached)
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="strict")

    def save(self, path=MERGES_FILE):
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": [[a, b] for (a, b), _ in ordered]}, f)


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def load(path=None):
    """Return the tokenizer used by train.py and evaluate.py (no args)."""
    p = path or MERGES_FILE
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
        if data.get("type") == "bpe":
            return BPETokenizer([tuple(m) for m in data["merges"]])
    return ByteTokenizer()   # safe fallback


# --------------------------- training -----------------------------------
def train_bpe(text, vocab_size):
    """Efficient incremental BPE over unique whitespace/word pieces."""
    from collections import Counter, defaultdict
    freq = Counter(_SPLIT.findall(text))
    # corpus[i] = [list_of_ids, count]; words are byte sequences
    corpus = [[list(w.encode("utf-8")), c] for w, c in freq.items()]

    pair_counts = Counter()
    pair_where = defaultdict(set)   # pair -> set of word indices containing it
    for i, (ids, c) in enumerate(corpus):
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += c
            pair_where[(a, b)].add(i)

    merges = []
    next_id = 256
    target = vocab_size - 256
    while len(merges) < target and pair_counts:
        # best pair by count, deterministic tie-break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best] <= 0:
            break
        a, b = best
        merges.append([a, b])
        involved = list(pair_where[best])
        for i in involved:
            ids, c = corpus[i]
            # subtract this word's current pairs
            for x, y in zip(ids, ids[1:]):
                pair_counts[(x, y)] -= c
            # merge a,b -> next_id
            new = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == a and ids[j + 1] == b:
                    new.append(next_id)
                    j += 2
                else:
                    new.append(ids[j])
                    j += 1
            corpus[i][0] = new
            # add the word's new pairs
            for x, y in zip(new, new[1:]):
                pair_counts[(x, y)] += c
                pair_where[(x, y)].add(i)
        # clean up dead entries
        for x, y in list(pair_counts):
            if pair_counts[(x, y)] <= 0:
                del pair_counts[(x, y)]
                pair_where.pop((x, y), None)
        next_id += 1
    return merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--out", default=MERGES_FILE)
    args = ap.parse_args()
    text = open(args.data, encoding="utf-8").read()
    print(f"training BPE: {len(text.encode('utf-8')):,} bytes -> vocab {args.vocab}")
    merges = train_bpe(text, args.vocab)
    tok = BPETokenizer([tuple(m) for m in merges])
    tok.save(args.out)
    # sanity: losslessness + compression on a sample
    sample = text[:200000]
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, "BPE round-trip failed!"
    nb = len(sample.encode("utf-8"))
    print(f"vocab_size={tok.vocab_size}  saved {args.out}")
    print(f"sample: {nb:,} bytes -> {len(ids):,} tokens  "
          f"({nb/len(ids):.2f} bytes/token, byte tok = 1.00)")


if __name__ == "__main__":
    main()
