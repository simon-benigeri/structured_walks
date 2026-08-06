# Follow-Up Work: From Synthetic Graphs to Practical Agent Intelligence

## Premise

Phase 1 of our research (the current plan) establishes a mechanistic finding in a controlled synthetic setting:

> LLMs form factored in-context representations that separate static relational structure (the graph) from dynamic traversal strategy (the walk).

This follow-up plan asks: **what does this buy you in practice?** We use RAGEN's interactive agent environments (FrozenLake, Sokoban) as a bridge from the synthetic finding to practical implications for LLM agents, dialogue systems, and long-context applications.

---

## The Gap Between Synthetic and Practical

| What we show in Phase 1 | What practitioners care about |
|---|---|
| Grid/ring geometry emerges in PCA | Does the agent actually navigate better? |
| Walk type is separable from graph topology | Can you control agent behavior by intervening on strategy representations? |
| Phase transition at N tokens | How much exploration/context does a real agent need before it "knows" its environment? |
| Multiple walk types don't destroy graph knowledge | Can an agent switch strategies without forgetting the environment? |

The follow-up work closes these gaps by running analogous experiments in environments where success is measurable by task reward, not just representational geometry.

---

## Why RAGEN

[RAGEN](https://github.com/mll-lab-nu/RAGEN) (mll-lab-nu, Northwestern) provides:

- **Grid-based environments** (FrozenLake, Sokoban) that are literally graph navigation problems — the same domain as our synthetic setup, but with goals, obstacles, and stochastic dynamics.
- **Real agent trajectories** at various stages of RL training (random exploration → competent policy).
- **Trained model checkpoints** whose internal representations we can probe.
- **Infrastructure for multi-turn interaction** where we can test whether in-context structure learning actually improves downstream performance.
- **Reasoning collapse diagnostics** (RAGEN-2) that may connect to our representation factoring findings.

Since RAGEN is from our own lab, access and collaboration are straightforward.

---

## Follow-Up Experiments

### Experiment A: Naturalistic Trajectories → Frozen LLM

**Question:** Does the in-context graph-structure phenomenon hold when the "walks" are real agent trajectories rather than synthetic random walks?

**Setup:**
1. Collect FrozenLake trajectories from RAGEN at different training stages:
   - Random policy (epoch 0) — equivalent to random walk
   - Mid-training (partially learned) — biased exploration
   - Converged policy (near-optimal) — highly structured, repetitive paths
2. Format these trajectories as token sequences (state descriptions or grid coordinates as concept tokens).
3. Feed them to a frozen Llama-3.1-8B. Extract activations. Run our full analysis pipeline (PCA, Dirichlet energy, accuracy).

**What we'd learn:**
- Does a random policy produce the same clean phase transition as a synthetic random walk? (Expected: yes, this is the direct bridge.)
- Does a *trained* policy's trajectories still reveal the grid structure, or does the low-entropy exploitation path give too little structural information? (This tells us about the information-structure tradeoff in exploration.)
- How does trajectory quality (random vs. trained) affect the *speed* of structure inference?

**Practical implication:** Tells you whether an agent's exploration history gives other models (or itself, in a future context window) enough signal to infer the environment map.

---

### Experiment B: Probing RAGEN-Trained Agents for Factored Representations

**Question:** Do RAGEN-trained agents internally separate "map knowledge" from "policy/strategy"?

**Setup:**
1. Take RAGEN model checkpoints at various stages of training (on FrozenLake or Sokoban).
2. Feed environment states to each checkpoint and extract activations.
3. Apply our analysis toolkit:
   - PCA: does the grid/map geometry emerge in the trained model's representations?
   - Linear probes: can we separate "where am I on the map" from "what policy am I following"?
   - Dirichlet energy on the map graph: does it decrease over training?

**What we'd learn:**
- Does RL training produce the same factored representation that in-context learning produces?
- Does RAGEN-2's "reasoning collapse" correspond to a collapse of this factoring? (If map and policy representations merge, the agent might produce template responses that ignore the current state — exactly what reasoning collapse looks like.)
- At what point in training does the map representation stabilize?

**Practical implication:** A new diagnostic for agent training health. If Dirichlet energy of the environment map stops decreasing, the agent has learned the environment. If it's still high, the agent is still confused about its world model.

---

### Experiment C: In-Context Map Learning → Better Agent Performance

**Question:** If you give an agent enough context to in-context learn the environment map *before* it acts, does it perform better?

**Setup:**
1. Use RAGEN's FrozenLake environment with a frozen (non-RL-trained) LLM.
2. Condition A (cold start): Agent acts immediately in the environment.
3. Condition B (warm-up): Agent is first given a "warm-up" context — N tokens of exploration traces from the same environment (generated by a random policy). Then it acts.
4. Vary N across the phase-transition range identified in Phase 1.
5. Measure: task reward, successful navigation rate.

**What we'd learn:**
- Does in-context structure inference (our Phase 1 phenomenon) *actually improve downstream task performance*?
- Is there a critical amount of warm-up context below which it doesn't help and above which it does? (Should correspond to our phase transition point.)
- Does the *type* of warm-up trajectory matter (random vs. structured)? (Connects to our Exp 1 on walk-type diversity.)

**Practical implication:** Directly actionable for agent system design. Answers the question: "How much exploration history should you include in an agent's context window before asking it to act?"

---

### Experiment D: Strategy Steering via Representation Intervention

**Question:** Can you change an agent's behavior by intervening on the "strategy" subspace without disrupting its "map" knowledge?

**Setup:**
1. From Phase 1 (Exp 2), identify the subspace encoding walk-type/strategy.
2. In RAGEN's FrozenLake, observe a model navigating with strategy A (e.g., exploratory, visiting many states).
3. Intervene: patch the strategy-subspace activations to push toward strategy B (e.g., exploitative, beelining to goal).
4. Measure: does the agent change behavior (shift from exploration to exploitation) without becoming confused about the map (no increase in collision with walls, no revisiting invalid states)?

**What we'd learn:**
- Is the factored representation *causally functional* — can you actually steer behavior by manipulating one factor while leaving the other intact?
- What are the limits? How large an intervention can you make before the map representation gets corrupted?

**Practical implication:** A mechanism for controllable agents. Instead of re-prompting or fine-tuning, you intervene on a specific activation subspace to change strategy. This is the agent analog of "steering vectors" for dialogue style.

---

### Experiment E: Reasoning Collapse as Representation De-Factoring

**Question:** Is RAGEN-2's "reasoning collapse" phenomenon caused by a collapse of factored representations?

**Setup:**
1. Train RAGEN agents on Sokoban with and without the SNR-Adaptive Filtering intervention.
2. At regular checkpoints, probe for factored representations (Experiment B's methodology).
3. Track two quantities over training:
   - RAGEN-2's mutual information diagnostic (I(X;Z) — are responses input-dependent?)
   - Our Dirichlet energy / probe separability (is map separate from policy?)
4. Correlate: when MI drops (reasoning collapse), does representation factoring also collapse?

**What we'd learn:**
- A mechanistic explanation for reasoning collapse: it happens when the model's internal structure stops distinguishing "what world am I in" from "what template do I use."
- Whether SNR-Adaptive Filtering works *because* it preserves representation factoring.

**Practical implication:** A deeper diagnostic and potentially a better intervention for reasoning collapse — directly maintain representation factoring rather than filtering by reward variance.

---

## Connection Map

```
Phase 1 (Synthetic)                    Follow-Up (RAGEN Bridge)
─────────────────────                  ──────────────────────────
                                       
Exp 0: Reproduction ─────────────────► Exp A: Same phenomenon on real trajectories
                                       
Exp 1-2: Multiple walks + probing ───► Exp B: Probe RL-trained agents
                                       Exp D: Steer via strategy subspace
                                       
Exp 4: Novel walk mid-context ───────► Exp C: Warm-up context helps agents?
                                       
Exp 5: Scaling laws ─────────────────► Exp C: Critical warm-up threshold
                                       
Theory: Static/dynamic factoring ────► Exp E: Reasoning collapse = de-factoring?
```

---

## Requirements Beyond Phase 1

| Need | Details |
|---|---|
| GPU compute for RL training | RAGEN needs multi-GPU (4-8x A100) for training runs. Probing frozen checkpoints is cheap. |
| RAGEN codebase access | Already in-lab (mll-lab-nu). Clone and use FrozenLake/Sokoban environments. |
| RAGEN checkpoints | Either train ourselves or get checkpoints from the RAGEN team. |
| NNsight compatibility with RAGEN models | RAGEN uses veRL-trained models. Confirm activation extraction works on their checkpoints. |

**Estimated additional compute:**
- Experiment A: ~5 GPU-hours (inference only, same as Phase 1)
- Experiment B: ~10 GPU-hours (inference on multiple checkpoints)
- Experiment C: ~20 GPU-hours (multi-turn interaction, multiple conditions)
- Experiment D: ~10 GPU-hours (intervention experiments)
- Experiment E: ~40+ GPU-hours (RL training runs with diagnostics)

Experiments A-D are tractable as a follow-up. Experiment E is a larger commitment and might be its own paper.

---

## Timeline (After Phase 1)

| Period | Work |
|---|---|
| Weeks 1-2 | Experiment A: collect RAGEN trajectories, run through our pipeline, confirm the bridge |
| Weeks 3-4 | Experiment B: probe RAGEN checkpoints for factored representations |
| Weeks 5-6 | Experiment C: warm-up context experiment in FrozenLake |
| Weeks 7-8 | Experiment D: causal intervention on strategy subspace |
| If continued | Experiment E: reasoning collapse connection (larger effort) |

---

## What This Gets Us (Publication Strategy)

**Paper 1 (Phase 1 alone):** "Structured Walks: LLMs Factor Static and Dynamic Context In-Context." Clean synthetic result, mechanism + theory. Target: ICLR/NeurIPS interpretability track.

**Paper 2 (Phase 1 + Follow-Up A-D):** "From Graphs to Agents: In-Context Structure Learning Predicts Agent Performance." Synthetic mechanism + practical validation on RAGEN environments. Target: ICML/NeurIPS main track (stronger applied story).

**Paper 3 (Experiment E, if results are strong):** "Reasoning Collapse as Representation De-Factoring." Mechanistic explanation of a known training pathology. Could be a short paper or merged into Paper 2.

The follow-up work transforms a "here's an interesting phenomenon" paper into a "here's a phenomenon with direct consequences for how you build and train agents" paper. That's the difference between a workshop paper and a top venue.
