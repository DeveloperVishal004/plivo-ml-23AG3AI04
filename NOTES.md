# NOTES

1. **Best config:** GPT (4 layers, 4 heads, n_embd 160, block 128), weight-tied,
   trained 2000 steps with AdamW + warmup(150)+cosine, wd 0.1, grad-clip 1.0,
   peak LR 2e-3, batch 32, on a **byte-level BPE tokenizer (vocab 2048)** trained
   on the corpus only. **bpb 1.7999** vs baseline 2.3718 (−24%), 1.59M params.
2. **Why it works — two independent levers.** The baseline was both
   *under-optimized* (constant LR, plain Adam, no clip) and *badly tokenized*.
3. Fixing the optimizer/schedule/init/tying alone took bpb 2.3718 → 2.0109.
4. The BPE tokenizer took it 2.0109 → 1.7999: the corpus is 33% Devanagari
   **bytes**, and byte-level tokens shred every Hindi character into 3 pieces.
5. BPE gives ~2.35 bytes/token, so each fixed step sees ~2× more real text and
   the window covers ~2× more characters — the Hindi part benefits most.
6. bpb is measured **per byte**, so the tokenizer win is real compression, not an
   accounting trick — decode(encode(x))==x is enforced by the scorer.
7. **Where it still fails:** vocab beyond ~1–2k gave almost nothing (rare tokens
   are under-trained in only 2000 steps), and longer context (block 256) *lost*
   once it forced a smaller batch — optimization noise dominated.
8. The remaining error is concentrated on rare Devanagari conjuncts and on the
   English↔Hindi script boundaries, where subword statistics are thin.
9. **With one more day:** a proper block-256 run at full batch 32, a 5th layer,
   and a BPE vocab tuned per-script (separate Hindi merges) — each tested cleanly.
10. Every number here is from the unmodified `evaluate.py`; ablations are one
    change at a time and logged in RUNLOG.md.
