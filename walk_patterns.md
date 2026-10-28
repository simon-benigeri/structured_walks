# Walk Patterns as Narrative and Conversational Structures

*Design note for discussion. Follows the Park et al. reproduction; inputs to
Experiments 1-4 of the research plan.*

## The idea

Park et al. show a model infers a graph from **uniform random walks** over it.
A uniform random walk is the least structured way to traverse a graph: every
step is independent of everything before it. Real language is not like that.
When a person explains, narrates, or argues, their movement through a space of
related topics follows recognizable procedures — they state a conclusion and
then justify it, they digress and return, they hold one idea fixed and compare
alternatives against it.

So the graph is the *subject matter* and the walk is the *discourse structure
being used to talk about it*. The reproduction establishes that models recover
the subject matter from the least informative possible traversal. The question
we want to ask is what happens when the traversal itself carries structure:
does the model still recover the graph, and does it separately represent the
procedure generating the sequence?

Each pattern below is a walk rule on the graph paired with the discourse
structure it is meant to stand in for.

## Two axes that organize the set

The patterns differ along two axes that we think matter more than the surface
descriptions:

**Locality** — does a consecutive pair in the sequence correspond to a real
edge? Local patterns hand the model the adjacency structure directly. Non-local
patterns only reveal it indirectly, through aggregation over many jumps or
through intermediate fill-in steps.

**Memory** — is the walk Markovian, or does generating it require state beyond
the current node? A digression that returns to its origin needs a stack. A
recap needs a history. These are the patterns where the walk itself is not
describable as a transition matrix, and where we would most expect a separable
"what am I currently doing" representation.

| Pattern | Discourse structure | Local | Markovian |
|---|---|---|---|
| Topic drift | free association, casual conversation | yes | yes |
| Expository sweep | step-by-step explanation, chronology | yes | yes |
| Elaboration spiral | circling a topic, restating with variation | yes | yes |
| Agenda drift | goal-directed interview, negotiation | yes | yes |
| Outline-then-fill | thesis then support, flash-forward | no | no |
| Pivot-and-branch | comparison, Socratic branching | yes | no |
| Digression-and-return | parenthetical aside, nested clause | yes | no |
| Recap loop | "so far we have covered...", lecture summary | no | no |
| Topic shift | interruption, changing the subject | no | yes |

## The patterns

### Topic drift (baseline)
**Structure.** Free association. Each utterance relates to the one before it,
with no larger plan.
**Walk.** Uniform random neighbor. This is the paper's protocol and our control.

### Expository sweep
**Structure.** Sustained explanation along one thread — "first this, then this,
then this." Chronological narration, procedural instructions.
**Walk.** Pick a direction; continue in it with high probability, turn rarely.
Long runs in a consistent heading.
**Why it is interesting.** Maximally predictable locally, but visits few nodes.
The model gets very strong evidence about a small part of the graph and little
about the rest, which should slow global recovery while sharpening local
structure.

### Elaboration spiral
**Structure.** Returning to a topic repeatedly from slightly different angles
before moving on. Common in teaching and in persuasive writing.
**Walk.** Traverse a neighborhood in a consistent rotational direction, working
outward.
**Why it is interesting.** High revisit rate within a region. Tests whether
repeated local exposure produces locally accurate geometry faster than uniform
sampling does.

### Agenda drift
**Structure.** Conversation with an objective — an interviewer steering toward
a topic, a negotiation converging on terms. Locally meandering, globally
directed.
**Walk.** Random walk with a directional bias.
**Why it is interesting.** Interpolates between topic drift and expository
sweep, so it gives us a tunable knob rather than a discrete condition.

### Outline-then-fill
**Structure.** State the destination, then supply the path to it. Thesis
followed by supporting argument; a narrative that opens at the ending and works
back; an abstract followed by a paper.
**Walk.** Jump to a node several hops away, then walk back filling in the
intermediate nodes.
**Why it is interesting.** This is the sharpest test of whether structure can
be inferred from traversals that are not locally faithful. The jump asserts a
relationship between two nodes that are *not* adjacent. If the model recovers
the graph anyway, it must be aggregating over the fill-in steps rather than
treating consecutive tokens as edges.

### Pivot-and-branch
**Structure.** Hold one idea fixed and explore alternatives from it. "Consider
X. One option is A... another is B..." Comparison, hypothetical reasoning,
Socratic questioning from a central claim.
**Walk.** Fix a node; take several short walks outward from it, returning to
the pivot between branches.
**Why it is interesting.** Produces a hub-and-spoke signature with one node
hugely over-represented. A model that recovers the true geometry must *not*
read visit frequency as centrality.

### Digression-and-return
**Structure.** The parenthetical aside. "The treaty — and I should mention the
earlier negotiations here — was signed in March." Nested subordinate clauses,
footnotes, interruptions that resolve.
**Walk.** Walk locally; occasionally push the current node onto a stack, wander
a few steps away, then retrace back to the pushed node and continue. Supports
nesting to depth greater than one.
**Why it is interesting.** We think this is the most linguistically
interesting pattern of the set, because it is the only one requiring a stack.
The return is fully determined by history the model would have to be
maintaining. If the model represents anything about "where I will return to,"
that is a representation of discourse state rather than of the graph, which is
exactly the static/dynamic separation we are after.

### Recap loop
**Structure.** "To summarize what we have covered so far." Periodic
consolidation in lectures, chapter summaries, meeting recaps.
**Walk.** Walk locally; periodically emit a subset of previously visited nodes
in visitation order, then resume from where you left off.
**Why it is interesting.** The recap segment is non-local and history
dependent. It also provides a clean behavioral probe: during a recap the
correct continuation is determined by history, not adjacency, so rule-following
accuracy should *drop* inside recaps if the model is tracking the graph and
*rise* if it is tracking the discourse procedure.

### Topic shift
**Structure.** Abrupt change of subject. Interruption, channel switching,
multi-threaded conversation.
**Walk.** With probability *p*, teleport to a uniformly random node, then
resume local walking. PageRank-style.
**Why it is interesting.** Directly dilutes the edge signal, so it gives us a
dose-response curve: how much non-local noise can the inference tolerate before
the graph stops emerging?

## Design question to settle before implementing

**How should a jump be rendered in the token sequence?** For the non-local
patterns, a jump emitted as a plain consecutive pair is indistinguishable from
an edge. The model would be receiving actively false adjacency evidence, and we
would be measuring robustness to label noise rather than the ability to infer
structure from non-local discourse.

Three options, and we should pick deliberately:

1. **Unmarked.** Emit jumps as ordinary consecutive tokens. Hardest condition;
   risks confounding the result with false-edge noise.
2. **Delimited.** Insert a separator token at discontinuities, the way a
   paragraph break or "anyway," marks a shift in prose. Keeps the adjacency
   signal clean and is arguably the more faithful analogue of real discourse,
   which does mark its transitions.
3. **Both, as a condition.** Treat marking as an independent variable. This is
   more runs but it answers a question of its own: can the model use an
   explicit discourse marker to discount misleading adjacency?

Our inclination is (3) for the outline-then-fill pattern specifically, since
that is where the false-edge problem is most acute, and (2) elsewhere.

## Measurement issue to control for

Walk patterns differ in how fast they expose distinct edges. The expository
sweep revisits the same edges constantly; topic drift covers the graph more
evenly; teleporting wastes a fraction of its steps on non-edges. Comparing
patterns at equal *context length* therefore conflates the structure of the
walk with sheer quantity of evidence.

We plan to report curves against both context length and **cumulative distinct
edges observed**, so that "this pattern makes inference slower" can be
separated from "this pattern simply shows the model less."

## What we expect to learn

- Whether the inferred graph is invariant to the procedure that generated the
  traversal, which would make graph structure a genuine statistical invariant
  rather than an artifact of uniform sampling.
- Whether the model separately represents which procedure is currently active,
  and if so whether that information is carried in different layers or
  different principal components than the topology.
- Whether patterns requiring memory — digression-and-return, recap — produce
  representations of discourse state distinct from representations of position.

## Open questions for discussion

1. Is the stack-based digression pattern worth prioritizing over the simpler
   local patterns? It is the most distinctive but also the furthest from the
   paper's setup.
2. The recap pattern gives a behavioral signature (accuracy should invert
   inside recap segments) without needing probes. Is that worth building first
   as a cheap sanity check on whether walk type is represented at all?
3. Should walk type switch at restarts only, or mid-sequence? Mid-sequence
   switching is more realistic for conversation but makes segment labels noisy
   for probing.
