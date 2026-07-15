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
    ap.add_argument("--opt", default="adamw")              # adamw | muon
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--qk_norm", type=int, default=0)
    ap.add_argument("--softcap", type=float, default=0.0)
    ap.add_argument("--mom_warmup", type=int, default=0)   # Muon momentum 0.85->0.95
    ap.add_argument("--zloss", type=float, default=0.0)    # PaLM z-loss coefficient
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
    if args.qk_norm:             cfg.qk_norm = True
    if args.softcap > 0:         cfg.logit_softcap = args.softcap

    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (cfg: L{cfg.n_layer} H{cfg.n_head} "
          f"E{cfg.n_embd} block{cfg.block_size} tie{int(cfg.tie_weights)})")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    # Optimizers. Each entry is (optimizer, base_peak_lr); the cosine schedule
    # scales every optimizer's LR by the same factor each step.
    opts = []
    if args.opt == "muon":
        from muon import Muon
        # hidden 2-D matrices -> Muon; embeddings / tied head / norms / biases -> AdamW
        muon_p, adamw_p = [], []
        for pn, p in model.named_parameters():
            if ("blocks" in pn) and p.dim() >= 2:
                muon_p.append(p)
            else:
                adamw_p.append(p)
        opts.append((Muon(muon_p, lr=args.muon_lr, momentum=0.95), args.muon_lr))
        opts.append((torch.optim.AdamW(adamw_p, lr=args.lr, betas=(0.9, 0.95),
                                       weight_decay=0.0), args.lr))
        print(f"optimizer: Muon on {len(muon_p)} matrices + AdamW on "
              f"{len(adamw_p)} tensors")
    else:
        decay, no_decay = [], []
        for pn, p in model.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        opts.append((torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.wd},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95)), args.lr))
        print("optimizer: AdamW")

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        frac = lr_at(step, args.steps, 1.0, args.warmup, args.min_lr_frac)
        for opt, base in opts:
            for g in opt.param_groups:
                g["lr"] = base * frac
                if args.mom_warmup and "momentum" in g:  # Muon momentum warmup
                    w = min(1.0, step / max(1, args.steps // 10))
                    g["momentum"] = 0.85 + (0.95 - 0.85) * w
        lr = args.lr * frac
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        logits, loss = model(x, y)
        if args.zloss > 0:   # PaLM z-loss: keep logsumexp(logits) near 0
            loss = loss + args.zloss * (torch.logsumexp(logits, dim=-1) ** 2).mean()
        for opt, _ in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for opt, _ in opts:
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
