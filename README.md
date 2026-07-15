# plivo-ml-23AG3AI04 — 2,000-Step LLM Speedrun

Make a tiny GPT trained from scratch as good as possible (lowest **bits-per-byte**
on held-out mixed English+Hindi text) under fixed caps: **≤2000 steps, ≤2M
params, CPU only, `train_corpus.txt` only, pure PyTorch — no pretrained weights.**

## Result

| | bpb (lower is better) |
|---|---|
| Baseline (as given) | 2.3718 |
| **This submission** | **1.6577** |
| **Reduction** | **−30.1%** |

Final model: 1,766,640 params (< 2M), 2000 steps, CPU. Full ablation (10 logged
runs, 3 honest negatives) in [RUNLOG.md](RUNLOG.md); short rationale in
[NOTES.md](NOTES.md); visual writeup in `SUMMARY.html`.

## What was changed vs. the baseline
1. **Optimizer/schedule** — AdamW + warmup→cosine LR + grad-clip + weight tying + scaled init.
2. **Tokenizer** — corpus-trained **byte-level BPE (vocab 2048)**, lossless with raw-byte fallback. The corpus is 33% Devanagari *bytes*; BPE stops the byte tokenizer from shredding every Hindi character into 3 tokens.
3. **Architecture** — Llama-style: **RoPE**, **RMSNorm**, **SwiGLU**, **QK-Norm**, 5 layers.
4. **Optimizer** — **Muon** (Newton-Schulz orthogonalised momentum) on the weight matrices + AdamW on embeddings/head/norms.

## Files
| file | purpose |
|---|---|
| `ckpt.pt` | final checkpoint (contains config + step count) |
| `evaluate.py` | official scorer (unmodified interface) |
| `tokenizer.py` | byte-level BPE; `load()` reads `bpe.json` |
| `bpe.json` | trained BPE merges |
| `model.py` | GPT (RoPE/RMSNorm/SwiGLU/QK-Norm) |
| `train.py` | trainer (AdamW/Muon, schedules, flags) |
| `muon.py` | Muon optimizer |
| `RUNLOG.md`, `NOTES.md`, `SUMMARY.html` | required writeups |

## How to score this checkpoint
```bash
python evaluate.py --checkpoint ckpt.pt --text_file <any_text_file>
# -> {"bpb": 1.6577, "n_params": 1766640, "steps": 2000, ...}
```

## How to reproduce from scratch
```bash
python tokenizer.py --data <corpus> --vocab 2048          # trains bpe.json
python train.py --data <corpus> --steps 2000 --n_layer 5 \
    --opt muon --muon_lr 0.02 --qk_norm 1 --out ckpt.pt   # ~7 min CPU
python evaluate.py --checkpoint ckpt.pt --text_file <corpus_or_dev>
```

## References (techniques used, all pure-PyTorch)
- RoPE — Su et al., *RoFormer*, 2021
- RMSNorm — Zhang & Sennrich, 2019
- SwiGLU — Shazeer, *GLU Variants*, 2020
- Muon + QK-Norm — K. Jordan, *modded-nanoGPT* (2024); *A Field Guide to NanoGPT Speedrun Optimizations*
