# Structured Walks: How LLMs Learn Dynamic Context from Walk Structures on Graphs

## Motivation

Park et al. (2025) ([ICLR: In-Context Learning of Representations](https://arxiv.org/abs/2501.00070)) showed that when an LLM is given random-walk traces over a graph whose nodes are familiar words (e.g. *apple*, *bird*), the model's internal representations reorganize—suddenly and emergently—to mirror the graph's geometry. The underlying graph structure (a grid, ring, or hexagonal lattice) is recovered purely from in-context observation, without any fine-tuning.

That work used **simple, uniform random walks**. We propose to extend the paradigm by introducing **multiple distinct walk structures**—structured traversal strategies that sit *on top of* the same underlying graph. This separates two levels of latent structure the model must infer:

1. **Static structure** — the graph topology (which nodes are neighbors).
2. **Dynamic structure** — the walk strategy currently governing the sequence (spiral, linear sweep, long-distance jumps, etc.).

This is analogous to a core distinction in computation and in natural language: static knowledge (a finite-state graph of relationships) versus dynamic, procedural context (the program or narrative strategy currently being executed over that knowledge). We want to know whether LLMs can disentangle and represent both.

---

## Background: What the Paper Established

| Finding | Detail |
|---|---|
| **In-context representation reorganization** | With enough context tokens, LLM representations (measured via PCA on residual-stream activations) snap into the geometry of the underlying graph. |
| **Emergent phase transition** | Accuracy on next-node prediction shows a two-phase curve: a slow phase followed by a sharp inflection. The transition point scales as a power law with graph size. |
| **Energy minimization** | The reorganization is well described by Dirichlet energy minimization over the graph Laplacian; when energy drops, PCA recovers the spectral embedding of the graph. |
| **Semantic priors can coexist** | When nodes have pre-existing semantic correlations (e.g. days of the week), the in-context graph structure appears in later principal components, subordinate to but coexisting with the pretrained structure. |
| **Not mere memorization** | Model accuracy curves cannot be explained by a simple "copy from context" baseline; the model is inferring global structure. |

---

## Part 0: Reproducing the Original Paper

Before extending the paradigm, we need a working reproduction of Park et al.'s core results. This gives us (a) confidence that we can replicate the phenomena, (b) a validated codebase to build our extensions on, and (c) direct familiarity with the analysis tools.

> **Note on existing code.** A [third-party repo](https://github.com/ai-in-pm/ICLR) exists but does not faithfully reproduce the paper. It uses OpenAI API embeddings and manual gradient steps rather than extracting internal activations from a pretrained LLM. We need to build the pipeline from scratch using NNsight.

### Step 0.1 — Infrastructure: Activation Extraction via NNsight

**Goal.** Set up a pipeline that feeds token sequences into Llama-3.1-8B and extracts residual-stream activations at every layer.

**Details:**
- Install [NNsight](https://github.com/ndif-team/nnsight) (Python library for transparent access to model internals via NDIF remote inference, or local inference with HuggingFace models).
- Load Llama-3.1-8B (either locally with sufficient GPU memory, or remotely via NDIF).
- Write a function: given a token sequence of length *N*, return a tensor of shape `(N, L, d)` — activations for each token at each of the *L* layers, with hidden dimension *d* = 4096.
- Verify by extracting activations for a short test sequence and confirming shapes and value ranges are sensible.

**Key dependencies:** `nnsight`, `torch`, `transformers` (for tokenizer).

### Step 0.2 — Graph Construction and Random Walk Generation

**Goal.** Implement the data-generating process from the paper.

**Graph types to implement:**

| Graph | Construction | Nodes |
|---|---|---|
| Square grid | `m × m` grid, edges between horizontal/vertical neighbors. No periodic boundaries. | 16 (4×4) or 25 (5×5) |
| Ring | *n* nodes in a cycle, edges between adjacent nodes. | 10 or 50 |
| Hexagonal lattice | Honeycomb grid. | 30 |

**Node labeling:** Assign each node a single-token word with no obvious semantic correlations (e.g. *apple*, *sand*, *math*, *river*, *chair*, ...). The paper uses words that tokenize to a single token in Llama's tokenizer — verify this.

**Walk generation:**
- **Grid/hexagonal:** Uniform random walk — at each step, move to a uniformly random neighbor. Emit the visited node's label. Sequences of ~2000–8000 tokens.
- **Ring:** Sample random neighboring pairs (not a walk per se, but random edge sampling). Emit pairs sequentially.

**Restarts:** Optionally restart from a uniformly random node after some number of steps (the paper mentions occasional restarts).

**Output format:** A flat list of token strings, e.g. `["apple", "sand", "chair", "sand", "river", ...]`.

### Step 0.3 — Mean Activation Computation and PCA Visualization

**Goal.** Reproduce Figures 1 and 2 from the paper — PCA plots showing representations reorganizing to mirror graph geometry.

**Procedure (following Section 3.1 of the paper):**
1. Feed the walk sequence into the model and extract residual-stream activations at a target layer (e.g. layer 26 of 32 for the grid; sweep multiple layers for the ring).
2. At each timestep *t*, look at a window of *N_w* = 50 preceding tokens.
3. For each unique concept $\tau$, compute the mean activation vector $\mathbf{h}_\tau^\ell$ by averaging all activations of $\tau$ within the window.
4. Stack mean vectors into a matrix $\mathbf{H}^\ell(\mathcal{T}) \in \mathbb{R}^{n \times d}$.
5. Run PCA on this matrix; plot the first two principal components, coloring/labeling each point by its node identity and grid position.

**What to check:**
- At short context lengths (few hundred tokens), representations should reflect pretrained semantics (e.g. *apple* and *orange* cluster together).
- At long context lengths (thousands of tokens), representations should snap into the graph's geometry (grid shape, ring shape).
- The transition should be sudden, not gradual.

### Step 0.4 — Dirichlet Energy Computation

**Goal.** Reproduce Figure 4 — Dirichlet energy as a function of context length, alongside accuracy.

**Dirichlet energy** for graph $\mathcal{G}$ with adjacency matrix $\mathbf{A}$ and representation matrix $\mathbf{H}$:

$$E_\mathcal{G}(\mathbf{H}) = \sum_{i,j} A_{i,j} \| \mathbf{h}_i - \mathbf{h}_j \|^2$$

**Procedure:**
1. Compute mean activations $\mathbf{H}^\ell(\mathcal{T})$ at multiple context lengths (e.g. every 100 tokens from 0 to 5000).
2. At each context length, compute $E_\mathcal{G}$ using the ground-truth adjacency matrix.
3. Plot energy vs. context length for several layers.

**Expected result:** Energy decreases as context grows, reaching a minimum shortly before accuracy begins its rapid ascent.

### Step 0.5 — Rule-Following Accuracy

**Goal.** Reproduce the accuracy curves from Figures 4 and 5.

**"Rule-following accuracy":** At each timestep, look at the model's output distribution over the vocabulary. Sum the probabilities assigned to all tokens that are valid neighbors of the current node in the graph. This is the accuracy for that step.

**Procedure:**
1. At each timestep in the sequence, record the model's next-token probability distribution.
2. Identify the current node (the most recent token).
3. Look up its neighbors in the ground-truth graph.
4. Sum the probabilities of those neighbor tokens.
5. Average over timesteps within a sliding window; plot as a function of context length.

**What to check:**
- Two-phase accuracy curve: slow initial improvement, then a sharp inflection.
- The inflection point should roughly coincide with where Dirichlet energy bottoms out.
- The memorization baselines (Equations 4–5 in the paper) should *not* explain the curve shape.

### Step 0.6 — Semantic Prior Experiment (Days of the Week)

**Goal.** Reproduce Figure 3 — when nodes are days of the week, pretrained semantics and in-context graph structure coexist in different principal components.

**Procedure:**
1. Use {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday} as node labels.
2. Randomly permute their ordering on a 7-node ring (so the ring order ≠ the calendar order).
3. Generate walk sequences and extract activations as before.
4. PCA on the mean activations:
   - PCs 1–2 should show the *pretrained* circular ordering (calendar order).
   - PCs 3–4 should show the *in-context* ring ordering (the permuted graph).

**Why this matters for our extensions:** This establishes the baseline for how pretrained structure and in-context structure interact — our Experiment 2 (walk-type probing) will build on exactly this kind of factored representation.

### Step 0.7 — Phase Transition Scaling

**Goal.** Reproduce Figure 8 — the accuracy inflection point scales as a power law with graph size.

**Procedure:**
1. Run the full pipeline (walk generation → activation extraction → accuracy computation) for square grids of size 3×3, 4×4, 5×5, 6×6.
2. For each, fit a piecewise linear function to the accuracy curve (log-linear) and extract the transition point.
3. Plot transition point vs. *m* (grid side length) on a log-log plot.
4. Fit a power law: transition point ~ $m^\alpha$.

**Expected result:** Clean power-law scaling, consistent with the bond-percolation hypothesis from the paper.

### Reproduction Milestones and Go/No-Go

| Milestone | Criterion | Go / No-Go |
|---|---|---|
| **0.1** Activation extraction works | Correct tensor shapes, reasonable value ranges | Must pass before anything else |
| **0.3** PCA shows grid/ring geometry | Visual match to Figures 1 & 2 | Core phenomenon. If this fails, debug before proceeding. |
| **0.4–0.5** Energy drops, accuracy rises | Qualitative match to Figure 4 curves | Quantitative machinery works. Proceed to extensions. |
| **0.6** Semantic prior separation | Calendar ring in PCs 1–2, graph ring in PCs 3–4 | Confirms factored representations. Critical for our Q1–Q3. |
| **0.7** Power-law scaling | Linear trend on log-log plot | Nice to have. Not blocking for our extensions. |

### Models

We run all experiments on three model families to establish architecture-generality:

| Model | Params | Architecture | Role |
|---|---|---|---|
| **Llama-3.1-8B** | 8B | GQA, 32 layers, RoPE | Primary (direct comparison with Park et al.) |
| **Gemma-2-9B** | 9B | Sliding window attention, different tokenizer | Different pretraining, validates generality |
| **Qwen-2.5-7B** | 7B | GQA, different tokenizer | Bridges to RAGEN follow-up (which uses Qwen) |

All three are available on HuggingFace, supported by NNsight, and similar in scale (~7–9B parameters). The pipeline is model-agnostic once activation extraction attribute paths are configured per architecture.

Smaller models (Llama-3.2-1B, Gemma-2-2B) can be tested as optional scale checks if time permits.

### Estimated Compute Requirements

- **Models:** Llama-3.1-8B, Gemma-2-9B, Qwen-2.5-7B (~16GB each in float16). Can run on a single A100/H100, or use NDIF remote inference for Llama.
- **Per run:** A single sequence of 5000 tokens through an 8B model with full activation extraction takes ~30–60 seconds on an A100.
- **Total for reproduction (all 3 models):** ~150–300 runs. Roughly 3–6 GPU-hours.
- **Total for extensions (Experiments 1–5, all 3 models):** ~30–60 GPU-hours. Still very manageable on a single A100.

---

## Core Research Questions

### Q1 — Can the model separate static graph structure from dynamic walk structure?

When sequences are generated by *different* walk strategies over the *same* graph, does the model learn a single, shared representation of the graph topology, distinct from its representation of the current walk type?

### Q2 — How does walk-structure diversity affect the speed and quality of graph learning?

Does exposure to multiple walk strategies slow down or speed up recovery of the underlying graph? Measured per-token (controlling for total exposure to individual graph edges), does variety help, hurt, or make no difference?

### Q3 — Does the model form explicit representations of walk structure?

Beyond recovering the graph, does the model develop internal representations that distinguish which walk strategy is currently active? If so, where in the network (which layers, which principal components) does this information live?

### Q4 — How does the model handle "long-distance" and non-local walk strategies?

Natural language often involves non-local narrative moves: flashbacks, outlines-then-details, comparisons across distant topics. If we introduce walk strategies that jump across the graph (not just step to neighbors), can the model still infer the graph and the walk logic?

---

## Proposed Experiments

### Experiment 1: Multiple Walk Structures on a Single Grid

**Setup.** Use the same square grid (e.g. 4×4 or 5×5) from Park et al. Generate long sequences composed of segments, where each segment follows one of *k* distinct walk strategies, chosen at random at each restart.

**Walk strategies (examples):**

| Strategy | Description |
|---|---|
| *Random walk* | Uniform random neighbor selection (baseline, replicating the original paper). |
| *Tight spiral* | Start at a node, spiral outward visiting neighbors in a consistent rotational direction. |
| *Linear sweep* | Pick a direction (e.g. east); step in that direction with 95%+ probability, occasionally turning. Parameterize by sweep length before a forced turn. |
| *Biased drift* | Like a random walk but with a directional bias (e.g. 60% probability of moving right). |

**Measurements:**
- PCA of residual-stream activations (replicating Park et al.'s visualization pipeline).
- Dirichlet energy as a function of context length (does the graph still emerge?).
- Accuracy on next-node prediction, compared against the single-walk baseline.
- Per-token learning efficiency: normalize context length by unique edge exposures.

**What we hope to learn:** Whether the graph representation is robust to walk-structure variation, and whether recovery is faster or slower when the model must also disentangle walk types.

---

### Experiment 2: Probing for Walk-Structure Representations

**Setup.** Same as Experiment 1. After collecting activations, apply linear probes (logistic regression on frozen activations) to classify which walk strategy generated the current segment.

**Measurements:**
- Probe accuracy by layer: where in the network is walk-type information encoded?
- Compare with the principal components that encode graph structure (from Experiment 1). Do they overlap or separate?
- Examine whether walk-type information appears in earlier or later PCs than graph structure.

**What we hope to learn:** Whether the model represents walk structure at all, and if so, whether it is represented *separately* from the graph topology—i.e., a factored representation of static vs. dynamic context.

---

### Experiment 3: Long-Distance and Non-Local Walk Strategies

**Setup.** Introduce walk strategies that involve jumps across the graph, not just neighbor steps:

| Strategy | Description |
|---|---|
| *Outline-then-fill* | Jump to a distant node (e.g. 5–10 hops away), then walk back filling in intermediate nodes. Models "explain the endpoints, then fill in the middle." |
| *Pivot-and-branch* | Hold a node fixed; take short walks (2–4 steps) in 2–3 different directions from it, listed sequentially. Models "explore scenarios from a central point." |
| *Teleport-and-resume* | With some probability *p*, jump to a uniformly random node (like PageRank teleportation), then resume local walking. Models restarts and topic shifts. |

**Measurements:**
- Does the model still recover the graph topology from these non-local traces?
- Dirichlet energy curves: do they still show a clean phase transition, or is it noisier / delayed?
- Accuracy on next-node prediction for local vs. non-local segments separately.

**What we hope to learn:** Whether the model can infer graph structure from walk strategies that only *implicitly* reveal local connectivity (through intermediate fill-in steps or through statistical aggregation of many jumps). This connects to how LLMs might handle non-linear narrative structures.

---

### Experiment 4: Introducing New Walk Structures Mid-Training (Context)

**Setup.** Begin the context window with segments from only 1–2 walk strategies. Partway through, introduce a novel walk strategy the model has not yet seen in this context.

**Measurements:**
- Does graph-structure quality (Dirichlet energy) degrade when the new walk type appears?
- How quickly does the model adapt to the new walk type (probe accuracy for walk classification)?
- Does prior graph knowledge (from earlier walk types) transfer, accelerating recovery?

**What we hope to learn:** Whether an already-inferred graph representation is robust to encountering a new dynamic pattern over it—a basic test of compositionality between static and dynamic representations.

---

### Experiment 5: Scaling — Number of Walk Types × Graph Size

**Setup.** Systematically vary:
- Graph size: 3×3, 4×4, 5×5, 6×6 grids (9 to 36 nodes).
- Number of walk types: 1, 2, 4, 8.

For each combination, measure the context length required to reach the accuracy inflection point (the "phase transition" from Park et al.).

**Measurements:**
- Transition point as a function of (graph size, number of walk types).
- Does the power-law scaling from the original paper still hold?
- Is there an additive or multiplicative cost to having more walk types?

**What we hope to learn:** The scaling relationship between structural complexity (graph size), procedural complexity (number of walk types), and the amount of context needed for in-context structure learning.

---

## Implementation Notes

**Why this is tractable:**

- **No benchmark creation needed.** All data is synthetically generated from simple graph + walk-strategy definitions.
- **Reproduction-first approach.** Part 0 builds all the core analysis tools (activation extraction, PCA, Dirichlet energy, accuracy curves). Once validated against the paper's results, these tools are reused directly for Experiments 1–5.
- **Small graphs suffice.** The original paper gets clean results with 10–36 node graphs and context windows of a few thousand tokens. Our extensions add walk-structure variation but don't require larger graphs.
- **Modest compute.** ~2 GPU-hours for reproduction, ~10–20 GPU-hours for all extensions. Single A100 or NDIF remote access suffices.

**Models:** Llama-3.1-8B (primary, matches paper), Gemma-2-9B, and Qwen-2.5-7B (for architecture-generality and RAGEN follow-up compatibility). See Section "Models" above.

**Codebase architecture (built incrementally during Part 0, extended in Parts 1–5):**

```
structured_walks/
├── graphs/
│   ├── grid.py              # Square grid construction
│   ├── ring.py              # Ring/cycle construction
│   └── hexagonal.py         # Honeycomb lattice construction
├── walks/
│   ├── random_walk.py       # Uniform random walk (reproduction)
│   ├── spiral.py            # Tight spiral strategy (Exp 1)
│   ├── linear_sweep.py      # Linear sweep strategy (Exp 1)
│   ├── biased_drift.py      # Biased drift strategy (Exp 1)
│   ├── outline_fill.py      # Outline-then-fill (Exp 3)
│   ├── pivot_branch.py      # Pivot-and-branch (Exp 3)
│   └── teleport.py          # Teleport-and-resume (Exp 3)
├── activation/
│   ├── extract.py           # NNsight activation extraction
│   └── mean_activations.py  # Windowed mean computation per concept
├── analysis/
│   ├── pca.py               # PCA visualization pipeline
│   ├── dirichlet.py         # Dirichlet energy computation
│   ├── accuracy.py          # Rule-following accuracy
│   ├── transition.py        # Piecewise-linear fit, transition detection
│   └── probes.py            # Linear probes for walk-type classification (Exp 2)
├── experiments/
│   ├── reproduce.py         # Part 0: full reproduction pipeline
│   ├── exp1_multi_walk.py   # Experiment 1
│   ├── exp2_probing.py      # Experiment 2
│   ├── exp3_nonlocal.py     # Experiment 3
│   ├── exp4_novel_walk.py   # Experiment 4
│   └── exp5_scaling.py      # Experiment 5
└── requirements.txt
```

**Key dependencies:** `nnsight`, `torch`, `transformers`, `numpy`, `scikit-learn`, `networkx`, `matplotlib`.

---

## Summary: What We Hope to Learn

| Question | Experiment | Expected Outcome |
|---|---|---|
| Does the graph emerge despite walk variation? | 1 | Yes — graph structure is a statistical invariant across walk types. |
| Is walk structure represented separately? | 2 | Walk-type info appears in different layers/components than graph topology. |
| Can non-local walks reveal local structure? | 3 | Yes, but with a delayed or noisier phase transition. |
| Is graph knowledge robust to novel walk types? | 4 | Partial — existing graph representation transfers but temporarily degrades. |
| How do graph size and walk complexity interact? | 5 | Transition point scales with both, possibly multiplicatively. |

The overarching thesis is that LLMs, in-context, can factor their representations into **what the world looks like** (graph topology) and **what is happening in the world right now** (walk strategy)—mirroring the static/dynamic distinction that pervades both computation and natural language understanding.
