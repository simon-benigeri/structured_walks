# Implementation Requirements

## Compute (Pick One Path)

| Option | What you need | Cost |
|---|---|---|
| **NDIF (free remote inference)** | Apply for access at [ndif.us](https://ndif.us). They provide remote Llama-3.1-8B with activation access via NNsight. No GPU needed locally. | Free (academic) |
| **Cloud GPU** | Single A100 (80GB) or H100 instance. ~20 GPU-hours total for reproduction + all experiments. | ~$30–60 on Lambda/RunPod/etc. |
| **Local GPU** | One 24GB+ GPU (e.g. RTX 3090/4090) can run 8B in float16 with some careful memory management, or use 4-bit quantization for activations. | Already owned |

NDIF is the path of least resistance — it's what the original paper used and it's free for researchers.

---

## Software Dependencies

```
torch>=2.0
transformers>=4.40
nnsight>=0.3
numpy
scikit-learn
networkx
matplotlib
scipy
```

All pip-installable. No exotic dependencies.

---

## Accounts / Access Needed

1. **NDIF access** — sign up at [ndif.us](https://ndif.us) (if going the remote inference route)
2. **HuggingFace account** — to download Llama-3.1-8B weights (requires accepting Meta's license, which is instant)
3. That's it. No OpenAI keys, no paid APIs.

---

## What to Build (in order of dependency)

| # | Component | Complexity | Est. Time |
|---|---|---|---|
| 1 | Activation extraction (`nnsight` wrapper) | Low — NNsight has good docs, ~50 lines | 1–2 hours |
| 2 | Graph construction (grid, ring) | Trivial — NetworkX one-liners | 30 min |
| 3 | Token vocabulary selection | Manual — pick ~36 single-token words, verify with tokenizer | 30 min |
| 4 | Random walk generator | Low — simple loop over graph neighbors | 30 min |
| 5 | Mean activation + PCA pipeline | Medium — windowed averaging, sklearn PCA, matplotlib | 2–3 hours |
| 6 | Dirichlet energy computation | Low — matrix multiply against adjacency | 30 min |
| 7 | Rule-following accuracy | Medium — need to capture logits + map to graph neighbors | 1–2 hours |
| 8 | Structured walk generators (spiral, sweep, etc.) | Medium — need to think about parameterization | 2–3 hours |
| 9 | Linear probes (Experiment 2) | Low — logistic regression on frozen activations | 1 hour |

**Total implementation time:** ~2–3 days of focused work for the reproduction (Steps 0.1–0.5). Another 2–3 days for the novel experiments.

---

## Estimated Compute Budget

| Phase | Runs | GPU-Hours |
|---|---|---|
| Reproduction (Part 0) | ~50–100 runs across graph sizes, context lengths, and layers | 1–2 |
| Extensions (Experiments 1–5) | ~5–10× reproduction | 10–20 |
| **Total** | | **~12–22 GPU-hours** |

A single sequence of 5000 tokens through Llama-3.1-8B with full activation extraction takes ~30–60 seconds on an A100.

---

## Key Risk / Blocker

The single biggest question is whether NDIF gives access to all 32 layers of residual-stream activations in a single forward pass for sequences up to ~8K tokens. If there are context-length or memory constraints on their end, we'd need to batch or go local. Worth confirming early.

---

## Timeline

| Week | Milestone |
|---|---|
| Week 1 | Steps 0.1–0.3: Activation extraction working, PCA shows grid/ring geometry |
| Week 2 | Steps 0.4–0.7: Dirichlet energy, accuracy curves, semantic prior experiment, scaling |
| Week 3 | Experiments 1–2: Multiple walk structures + probing |
| Week 4 | Experiments 3–5: Non-local walks, novel walk types, scaling analysis |
