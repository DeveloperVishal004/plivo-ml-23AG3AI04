# RUNLOG — 2,000-step LLM speedrun

Metric: **bits per byte (bpb)** on `dev_eval.txt`, via the unmodified
`evaluate.py`. Lower is better. All runs: CPU only, 2000 steps, < 2M params,
`train_corpus.txt` only. One change isolated per run.

| # | Change (vs previous) | bpb | n_params | notes |
|---|----------------------|-----|----------|-------|
| 0 | **Baseline** (given): byte tok, Adam lr=3e-4 constant, no wd/clip/warmup, tie=False, init std=0.05 | **2.3718** | 1,339,840 | our number to beat |
| 1 | Optimizer + init overhaul (byte tok kept) | **2.0109** | 1,298,880 | biggest single training fix |
| 2 | + BPE tokenizer, vocab 1024 | **1.8258** | 1,421,760 | the headline win |
| 3 | BPE vocab 2048 | **1.7999** | 1,585,600 | diminishing returns on vocab |
| 4 | + context 256 (batch 16, same tokens/step) | **1.9092** | 1,606,080 | **regressed** — see analysis |

**Final submitted model = Run 3: bpb 1.7999** (vocab 2048, block 128, batch 32).
That is a **24.1% reduction** over the 2.3718 baseline, within all caps.

---

## Run 0 — baseline
**Hypothesis:** none; establish the floor.
The starter is deliberately mediocre. Things I flagged as wrong before touching it:
constant LR with no warmup/decay, plain Adam with no weight decay, no gradient
clipping, `tie_weights=False` on a tiny model, a single init std=0.05 for every
tensor, and a **byte tokenizer** on a corpus that is 33% Devanagari *bytes*.
**Result:** bpb **2.3718**, 2000 steps, ~27 ms/step.

## Run 1 — optimizer + init + tying (tokenizer unchanged)
**Hypothesis:** most of the baseline's weakness is training dynamics, not capacity.
**Changed:** Adam→**AdamW** (betas 0.9/0.95, weight_decay 0.1 on 2-D weights only);
constant LR → **linear warmup (150) + cosine decay** to 10%, peak 2e-3; added
**grad-norm clip 1.0**; **weight tying** (head = input embedding); GPT-2 **init**
(std 0.02, residual projections scaled by 1/√(2·n_layer)); batch 8→32.
**Result:** bpb **2.0109** (−0.361). Train loss fell smoothly to ~1.36 vs the
baseline's ~1.73. Conclusion: the schedule + AdamW + clip is the single biggest
lever; the model was under-optimized, not under-parameterised.

## Run 2 — BPE tokenizer, vocab 1024
**Hypothesis:** the byte tokenizer wastes the tiny model on Devanagari byte
fragments (each Hindi char = 3 bytes = 3 tokens). Merging bytes into subwords
gives ~2× fewer tokens, so each of the fixed 2000 steps sees ~2× more real text
and the 128-token window covers ~2× more characters.
**Changed:** byte tok → **byte-level BPE (vocab 1024)**, trained on the corpus
only; lossless (decode(encode)==text verified by evaluate.py's round-trip).
**Result:** bpb **1.8258** (−0.185). Eval tokens 159,225 → 68,422 (2.09 B/tok).
Conclusion: confirmed — the tokenizer is the second big lever, and it helps
Hindi most.

## Run 3 — BPE vocab 2048
**Hypothesis:** more merges → more compression → lower bpb.
**Changed:** vocab 1024 → 2048.
**Result:** bpb **1.7999** (−0.026 only). Compression 2.09 → 2.35 B/tok, but the
rarer tokens are under-trained in just 2000 steps and params/step-time rose.
Conclusion: vocab past ~1–2k is diminishing returns at this step budget.

## Run 4 — longer context (block 256)
**Hypothesis:** with BPE the model can afford a wider window; more left-context
lowers next-token entropy, and `evaluate.py` scores with real left context too.
**Changed:** block_size 128 → 256, batch 32 → 16 (kept tokens/step ≈ 4096 so
step time is comparable).
**Result:** bpb **1.9092** — **worse** than Run 3 (1.7999). Train token-loss also
higher (~3.10). **Why it lost:** to keep tokens/step constant I halved the batch
(32→16), which doubled gradient-estimate variance across the fixed 2000 steps; the
longer 256-position embeddings are also seen less often and stay under-trained.
The optimization penalty outweighed the extra context. **Fix / conclusion:** revert
to Run 3's block 128 / batch 32. A clean context test would need block 256 at the
*full* batch 32 (≈2× compute) — left as future work given the CPU/time budget.

## Final
Chosen checkpoint = **Run 3** → `ckpt.pt`. bpb **1.7999**, 1,585,600 params,
2000 steps. Exact grader command verified inside the folder:
`python evaluate.py --checkpoint ckpt.pt --text_file <file>`. Tokenizer round-trip
verified lossless on English, Hindi, whitespace, and emoji.
