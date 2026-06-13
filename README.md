# Tiny GPT From Scratch 

I wanted to understand how GPT actually works — not at a "attention is query-key-value" level, but at a "here is every matrix multiply and its gradient" level. So I built one from scratch in pure NumPy.

No PyTorch. No autograd. Every forward pass, every backward pass, every attention head — written by hand.

---

## What's inside

| Component | Details |
|---|---|
| **Tokenizer** | Character-level vocab, encode/decode |
| **Bigram baseline** | Count-based + neural bigram models |
| **Embeddings** | Token + positional embeddings with backprop |
| **Attention** | Scaled dot-product, causal masking, softmax |
| **Multi-head Attention** | Head splitting, reshape, merge — fully from scratch |
| **Transformer Block** | Pre-LayerNorm, FFN, residual connections |
| **Stacked Blocks** | Forward + backward through N layers |
| **Optimizer** | Adam with bias correction, manual parameter updates |
| **Sampling** | Temperature scaling, top-k filtering, generation loop |

166 functions total — from `build_vocab` all the way to `decode_final_sequence`.

---

## Architecture
Input chars
│
▼
Token Embedding + Positional Embedding
│
▼
┌─────────────────────────┐
│   Transformer Block × N │
│  ┌───────────────────┐  │
│  │ Pre-LayerNorm     │  │
│  │ Multi-Head Attn   │  │
│  │ Residual Add      │  │
│  │ Pre-LayerNorm     │  │
│  │ FFN (ReLU)        │  │
│  │ Residual Add      │  │
│  └───────────────────┘  │
└─────────────────────────┘
│
▼
Final LayerNorm → LM Head → Softmax → Sample
---
## How to run
```bash
pip install numpy
python scaffold.py
```

---

## Things I learned building this

- **Causal masking** is simpler than it sounds — you just fill the upper triangle with -inf before softmax so future tokens get zero attention weight
- **Backprop through attention** was the most painful part — deriving gradients for Q, K, V projections by hand takes a while but once it works it feels like magic
- **Multi-head attention** is really just splitting the embedding dimension, running attention in parallel, then concatenating — the reshape and transpose order matters a lot
- **LayerNorm backward** has a non-obvious gradient — spent more time on this than I expected
- **Top-k sampling with temperature** — small temperature = confident and repetitive, high temperature = creative and chaotic. Makes sense once you see the softmax distribution change live
- Starting from a bigram model and incrementally adding complexity (embeddings → single head → multi-head → stacked blocks) made the whole thing much easier to debug

---

## Why NumPy only?

Every time I used PyTorch before, I trusted that `.backward()` was doing the right thing. Now I've derived and implemented those gradients myself — for attention, for LayerNorm, for residual connections. I know what's happening because I wrote it.

---

## Project structure

├── model.py       # All 166 functions implemented from scratch

├── scaffold.py    # Training loop, data pipeline, evaluation

└── README.md
---

*Built on [Deep-ML](https://www.deep-ml.com)*
