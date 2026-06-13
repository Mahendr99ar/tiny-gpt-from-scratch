# Tiny GPT From Scratch 🧠

A character-level GPT built **entirely in NumPy** — no PyTorch, no TensorFlow, no autograd. Every forward pass, backward pass, attention head, and Adam update is implemented manually from first principles.

---

## What's inside

This project builds a complete Transformer language model step by step:

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

**166 functions implemented** — from `build_vocab` all the way to `decode_final_sequence`.

---

## Architecture

```
Input chars
     │
     ▼
Token Embedding + Positional Embedding
     │
     ▼
┌─────────────────────────┐
│   Transformer Block × N  │
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
```

---

## How to run

```bash
# No dependencies beyond NumPy
pip install numpy

python scaffold.py
```

---

## Key concepts implemented

- **Causal self-attention** with masking (no future token leakage)
- **Backprop through attention** — manual gradients for Q, K, V projections
- **LayerNorm forward + backward** from scratch
- **Adam optimizer** — first/second moment estimates, bias correction
- **Top-k sampling** with temperature for text generation

---

## Why NumPy only?

Building this without any autograd framework forces you to derive and implement every gradient by hand. This gives a deep understanding of what PyTorch/JAX actually do under the hood.

---

## Project structure

```
├── model.py       # All 166 functions implemented from scratch
├── scaffold.py    # Training loop, data pipeline, evaluation
└── README.md
```

---

*Built on [Deep-ML](https://www.deep-ml.com) — Tiny GPT From Scratch project.*
