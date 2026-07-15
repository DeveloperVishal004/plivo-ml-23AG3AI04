# NOTES

1. **Best config:** Llama-style GPT — 5 layers, 4 heads, n_embd 160, block 128,
   **RoPE** positions, **RMSNorm**, **SwiGLU** MLP, **QK-Norm**, weight-tied —
   trained 2000 steps with **Muon** (2-D matrices, lr 0.02) + AdamW
   (embeddings/head/norms), warmup+cosine, on a corpus-trained **byte-level BPE
   (vocab 2048)**. **bpb 1.6577** vs baseline 2.3718 (**−30.1%**), 1.77M params.
2. **Why it works — three independent levers, each isolated in RUNLOG.**
3. *Tokenizer* is the structural win: the corpus is 33% Devanagari **bytes**, and
   the byte tokenizer shreds every Hindi char into 3 tokens. BPE (~2.35 B/tok)
   halves the sequence, so each fixed step sees ~2× more text; Hindi gains most.
4. *Architecture*: RoPE (no learned position table), RMSNorm, and SwiGLU are the
   modern Llama components — they lowered bpb **and** freed params (RoPE removed
   the position embedding), which paid for the 5th layer.
5. *Optimizer + attention*: Muon orthogonalises each momentum update
   (Newton-Schulz) on the weight matrices and beat AdamW under the step cap;
   **QK-Norm** (RMS-normalising q,k) then stopped attention from saturating and
   let Muon push harder (1.6855 → 1.6577).
6. bpb is **per byte**, so the tokenizer gain is real compression, not accounting;
   `decode(encode(x))==x` is enforced (verified on Hindi, code, emoji, tabs).
7. **Where it still fails / plateaus:** vocab past ~2k barely helps (rare tokens
   under-trained in 2000 steps); the model is now capacity-bound near the 2M cap,
   not step-bound; residual error concentrates on rare Devanagari conjuncts and
   script-boundary tokens.
8. **Honest negatives (kept, not hidden):** context-256 at half batch (noise beat
   context), logit soft-cap + momentum warmup (hurt calibration), and Muon lr
   0.03 (0.02 was better) all regressed — all logged in RUNLOG.
9. **With one more day:** block-256 at full batch 32, a per-script BPE, value
   embeddings, and z-loss — each tested one change at a time.
10. Every number is from the unmodified `evaluate.py` on `dev_eval.txt`.
