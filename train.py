"""Trainer (modified from the mediocre baseline).

Changes vs. baseline, each isolated in RUNLOG.md:
  * Adam(lr=3e-4, constant)  ->  AdamW(betas=(0.9,0.95), weight_decay)
  * no schedule              ->  linear warmup + cosine decay to min_lr
  * no gradient clipping      ->  clip grad norm to --clip
  * exposes batch/block/depth/width so the model + context can be tuned
  * records the full config + step count in the checkpoint (grader reads it)

HARD CAPS (unchanged, violation = disqualified):
  * <= 2000 optimizer steps      * <= 2,000,000 params
  * train_corpus.txt only        * pure PyTorch/numpy/stdlib, no pretrained

    python train.py --data ../llm_handout/data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, steps, peak, warmup, min_frac):
    if step < warmup:
        return peak * step / max(1, warmup)
    t = (step - warmup) / max(1, steps - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)      # peak LR
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--tie", type=int, default=None)       # 1/0 override
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    if args.block is not None:   cfg.block_size = args.block
    if args.n_layer is not None: cfg.n_layer = args.n_layer
    if args.n_embd is not None:  cfg.n_embd = args.n_embd
    if args.n_head is not None:  cfg.n_head = args.n_head
    if args.tie is not None:     cfg.tie_weights = bool(args.tie)

    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (cfg: L{cfg.n_layer} H{cfg.n_head} "
          f"E{cfg.n_embd} block{cfg.block_size} tie{int(cfg.tie_weights)})")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    # AdamW with weight decay only on 2-D weights (not norms/biases/embeddings).
    decay, no_decay = [], []
    for pn, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
