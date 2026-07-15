# NOTES

1. **Best config:** Llama-style GPT — 5 layers, 4 heads, n_embd 160, block 128,
   **RoPE** positions, **RMSNorm**, **SwiGLU** MLP, weight-tied — trained 2000
   steps with **Muon** (2-D matrices) + AdamW (embeddings/head/norms), warmup+
   cosine, on a corpus-trained **byte-level BPE (vocab 2048)**. **bpb 1.6855**
   vs baseline 2.3718 (**−28.9%**), 1.77M params (< 2M cap).
2. **Why it works — three independent levers, each isolated in RUNLOG.**
3. *Tokenizer* is the structural win: the corpus is 33% Devanagari **bytes**, and
   the byte tokenizer shreds every Hindi char into 3 tokens. BPE (~2.35 B/tok)
   halves the sequence, so each fixed step sees ~2× more text; Hindi gains most.
4. *Architecture*: RoPE (no learned position table), RMSNorm, and SwiGLU are the
   modern Llama components — they lowered bpb **and** freed params (RoPE removed
   the position embedding), which paid for the 5th layer.
5. *Optimizer*: Muon orthogonalises each momentum update (Newton-Schulz) on the
   weight matrices; under a hard 2000-step cap, better per-step updates matter
   more than anything, and it beat a well-tuned AdamW.
6. bpb is **per byte**, so the tokenizer gain is real compression, not accounting;
   `decode(encode(x))==x` is enforced (verified on Hindi, code, emoji, tabs).
7. **Where it still fails / plateaus:** vocab past ~2k barely helps (rare tokens
   under-trained in 2000 steps); the model is now capacity-bound near the 2M cap,
   not step-bound; residual error concentrates on rare Devanagari conjuncts and
   script-boundary tokens.
8. **Honest negative:** doubling context to 256 by halving the batch *regressed*
   (gradient noise beat the context gain) — kept in the log rather than hidden.
9. **With one more day:** block-256 at full batch 32, a per-script BPE, a Muon
   learning-rate sweep, and QK-norm — each tested one change at a time.
10. Every number is from the unmodified `evaluate.py` on `dev_eval.txt`.
