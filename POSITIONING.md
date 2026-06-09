# Alpha Evolver: what is actually new, and does it work

This note states the one defensible novelty claim, places it against the
literature, and reports the ablation and adversarial evidence for it. It is
deliberately conservative: most of the machinery is a recombination of known
ideas, and the point of the experiments is to find the parts that earn their keep.

## Max Cut creativity laboratory

Max Cut is now the default laboratory. The agent evolves reusable heuristic
programs across diverse graph families instead of tuning trading formulas. Each
program contains a static initializer, a dynamic vertex move scorer, and bounded
local search controls. The verifier is exact, deterministic, and hard limited.

The experiment separates four kinds of evidence:

- Train graphs determine primary search fitness.
- Development graphs verify generalization, trust, and prediction error.
- Sealed regular and heavy tail graph families are evaluated only after search.
- Replay graphs measure whether contextual memory causally changes discovery.

Before development verification, the agent predicts development quality from the
observed train result and the train to development gaps of similar prior
programs in matching contexts. This makes prediction error a generalization
surprise rather than another name for fitness.

Contextual memory stores immutable episodes and evidence grounded decision cards.
Cards require support from two independent run seeds before becoming trusted.
Retrieval includes the grounded contradiction episodes. Sealed graph identifiers
fail closed if they appear anywhere in memory.

The learned utility controller chooses dream, focus, challenge, recover, or
sleep every five generations. It cannot stop before generation 30, and external
limits always stop a run by 90 minutes or 250 generations.

Two tested negatives shaped the implementation:

- A global predictor assigned nearly the same prediction to every candidate and
  was rejected because surprise became a disguised quality score.
- An online ridge predictor received hundreds of observations but worsened live
  calibration, so it was removed instead of being shipped as extra machinery.

The retained contextual generalization predictor reduced mean absolute
prediction error from 0.0137 in the first five generations to 0.0087 in the last
five on a deterministic 30 generation calibration check. This is a mechanism
check, not yet a broad result.

### Max Cut surprise ablation

Fixed 100 generation searches, population 12, five matched seeds:

| variant | sealed quality |
|---|---|
| plain | 0.7380 ±0.0050 |
| monotonic surprise plus quarantine | 0.7379 ±0.0037 |
| quarantine only | 0.7366 ±0.0035 |
| inverted U plus quarantine | 0.7362 ±0.0047 |
| inverted U without quarantine | 0.7361 ±0.0039 |
| fixed greedy reference | 0.7264 |

The evolved programs transfer better than the fixed greedy reference. No
surprise reward beats plain search. Inverted U trails quarantine only by 0.0004
on average and wins three of five paired seeds, so it is not established.

The Max Cut default is therefore novelty plus quarantine with no surprise reward.
Inverted U remains an explicit experimental variant. Plain has the highest mean
by 0.0001 over monotonic surprise plus quarantine, but quarantine keeps its
independently demonstrated safety role at a small clean suite cost.

### Max Cut stopping comparison

Matched 100 generation ceilings, population 12, five seeds:

| policy | sealed quality | generations used |
|---|---|---|
| fixed | 0.7366 +/-0.0035 | 100.0 +/-0.0 |
| rules | 0.7378 +/-0.0037 | 55.0 +/-11.0 |
| learned | 0.7396 +/-0.0048 | 87.0 +/-11.2 |

Rule based stopping is the practical default. It retained sealed quality while
using 45 percent fewer generations. Learned stopping also retained quality, but
saved only 13 percent and had worse calibration. It remains experimental.

### Max Cut contextual memory replay

After building memory across five independent seeds, matched 12 generation cold
and warm replays produced mean lift of -0.00002, with positive lift in two of
five seeds. Warm memory changed every search trajectory, so the mechanism is
causally active, but it has not demonstrated a discovery benefit. We reject the
claim that contextual memory improves search until replay lift becomes positive.

## The original trading hypothesis

> **Productive surprise**: a single inverted-U prediction-error signal that
> *reinforces memory* for moderate surprise while *quarantining and stress-testing*
> extreme surprise as likely-toxic.

That was the original trading hypothesis: curiosity's inverted-U (the Wundt /
Berlyne / Kidd-Hayden curve) used as
**both** an exploration driver **and** a safety gate, in one signal. Around it sits
a deliberately boring "organism": dream / focus / recover modes under a thought-risk
budget, a quarantine trust-boundary, lineage circuit-breakers, and a sleep phase
that distils falsifiable causal hypotheses. The motto: *imagine recklessly, believe
cautiously, test relentlessly, remember with context.*

The Max Cut result does not support promoting this hypothesis to a general
mechanism. Quarantine transfers as a useful safety rule. The precise inverted U
reward does not.

## What is borrowed vs. new (honest)

Every **ingredient** has close prior art (verified via a paperclip literature scan):

| Mechanism | Closest prior art | Status |
|---|---|---|
| Predict performance before evaluating | Surrogate-assisted evolutionary computation | established |
| Surprise drives search | "Quality Diversity Through Surprise" (Gravina/Yannakakis 2018); Barto, "Novelty or Surprise?" (2013) | established (but monotonic, not inverted-U) |
| Inverted-U curiosity | "The Curious U" (Ten 2025); Wundt/Berlyne | established cognitive principle |
| Dream/focus dual process | DMN↔ECN creativity; "Generative System 3" (2025) | established |
| Reflective / causal memory | ExpeL; SRMA (2026) | established |
| Learning-progress curiosity | Oudeyer; Ten 2021; Poli 2022 | established |
| Domain (formulaic alpha mining) | a 2025 *survey* exists | crowded |

The **novelty is the specific fusion**, not any piece: using the inverted-U as a
dual-purpose reinforcement-and-safety signal, wrapped in a risk-budgeted
imagination layer with lineage containment and falsifiable causal memory, as one
compact numpy research organism. Honest scores: new *principle*
~2/10; novel *combination* ~5/10; compact *prototype* ~6/10; *publishable today*
~3/10 without the evidence below.

## Does it work? The ablation

Each variant isolates one mechanism (`--variant`; see `VARIANTS` in
`alpha_evolver.py`), run by `ablate.py` across seeds on identical panels.

- `plain` — fitness-only GP (no surprise, novelty, quarantine, modes, lineage)
- `novelty` — + novelty search
- `surprise_mono` — monotonic surprise (Gravina style)
- `surprise_noquar` — our inverted-U surprise but **no** quarantine
- `quar_mono` — quarantine but **monotonic** surprise (ablates the inverted-U)
- `with_lp` — full system plus the tested learning progress signal
- `ours` — everything

### Clean synthetic data (5 seeds, 60 generations)

| variant | best OOS Sharpe | discovery speed (gens to 90%) |
|---|---|---|
| plain | +1.97 ±0.29 | 19.8 |
| novelty | +2.00 ±0.30 | 28.8 |
| surprise_mono | +2.23 ±0.17 | 20.4 |
| surprise_noquar | +2.21 ±0.14 | 15.2 |
| quar_mono | +2.27 ±0.13 | 32.8 |
| **ours** | **+2.34 ±0.37** | **13.0** |

Read: the full system finds the best alphas **and** discovers fastest (90% of its
best in 13 generations vs 20–33). The ladder `plain < novelty < surprise-family <
ours` holds across 5 seeds. Effect size is real but modest (~19% over plain); the
discovery-speed gap is the clearer win.

### Adversarial data — the toxic-surprise test

`adversarial_panel` plants an **in-sample-only overfit trap** plus extreme jackpot
days: alphas that fit it look strong in-sample but die out-of-sample. Toxicity is
labelled **independently** by re-testing each trusted alpha on a *clean* panel.

| variant | overfit % of trusted memory | clean generalization of top alphas |
|---|---|---|
| plain (no quarantine) | 76% | −0.21 |
| surprise_noquar (no quarantine) | 76% | −0.23 |
| quar_mono (quarantine, monotonic) | 52% | +0.47 |
| **ours** (quarantine on) | **55%** | **+0.29** |

Read (5 seeds): the quarantine firewall cuts overfit junk in trusted memory from
**~76% to ~52–55%**, and — the sharper signal — its top alphas **generalize to
clean data (+0.29 to +0.47) while the no-firewall variants go negative (−0.21,
−0.23)**. Without the firewall the memory's "best" alphas are trap-fitted and fail
on fresh data; with it they hold up. This is the safety half of "productive
surprise" doing real work. Caveats: it does not *eliminate* overfit (the hardest
seeds stay high), and on this particular trap the monotonic+quarantine variant
edges out the inverted-U — so the inverted-U's *specific* advantage over plain
quarantine is not established here, only the value of quarantining at all.

### Learning progress (the brain-deepening addition) — tested, left off

Per the curiosity literature, human exploration tracks *learning progress* (where
am I improving?), which raw surprise and novelty miss. We added a per-operator
progress tracker (`Memory.progress_score`) that biases elite selection toward
operators whose outcomes are still rising. `--variant with_lp` turns it on.

Verdict (5 seeds): it does **not** help here. `with_lp` matches `ours` on best OOS
Sharpe (+2.34 either way) but **discovers slower** (~22 generations to 90% of best
vs 13) and with higher variance — it helps some seeds, hurts others, a net wash on
quality and a net loss on speed. That is the expected shape for a *convergent* task
with a findable optimum: learning-progress curiosity is built for open-ended search
where there is no single target to converge on. The mechanism is kept but **off by
default** — an honest negative, not an assumed win. (Two implementation notes that
cost real debugging: per-fragment histories were too sparse to fire, and the signal
first fed a vestigial `selection_score` that all elites bypass; it only became live
once it entered elite selection.)

## Limitations

- Synthetic data with planted structure. The headline robustness claim should be
  re-run on real or held-out market data before any strong statement.
- Modest effect sizes; more seeds are needed for tight confidence intervals.
- "Meaning" is operational (hypotheses + evidence + falsifiers), not semantic
  understanding.
- Surrogate-assisted EA is named as a baseline but not implemented (its value
  prop — skipping expensive evaluations — does not apply when backtests are cheap).

## What would raise it from "interesting" to "result"

1. More seeds + significance tests on the ladder above.
2. The adversarial test on real distribution shift, not just a synthetic trap.
3. Position publicly as the **dual-use inverted-U** mechanism, not "another alpha
   miner" or "another AI scientist" — both lanes are full.
