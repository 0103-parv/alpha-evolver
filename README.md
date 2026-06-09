# Alpha Evolver

Alpha Evolver is a self improving research agent. Its default laboratory evolves
reusable Max Cut heuristics across diverse graph families. It predicts before
testing, verifies every program in a bounded executor, stores grounded contextual
memory, and learns when another generation is likely to be useful.

The original trading signal laboratory remains available as the `trading`
adapter.

The default path is offline and uses only numpy plus the standard library. Optional integrations are guarded for Claude, Weave, yfinance, matplotlib, and Streamlit.

## First bounded run

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy matplotlib
./.venv/bin/python alpha_evolver.py --domain maxcut --generations 40 --pop 24
```

The default rule based controller may request a clean stop after generation 30.
The learned controller is available as an experimental policy. External limits
always stop the run by 90 minutes or 250 generations. The command writes:

- `graph_memory.json`
- `graph_history.csv`

At the end, matched 12 generation cold and warm replay searches report whether
contextual memory improved discovery.

## Run commands

```bash
# Default Max Cut laboratory
./.venv/bin/python alpha_evolver.py --domain maxcut --generations 100 --pop 24

# Fixed budget surprise experiment
./.venv/bin/python graph_ablate.py --experiment surprise --generations 100 --seeds 1 2 3 4 5

# Contextual memory lift experiment
./.venv/bin/python graph_ablate.py --experiment memory --generations 30 --pop 12 --seeds 1 2 3 4 5

# Learned versus rule versus fixed stopping
./.venv/bin/python graph_ablate.py --experiment stopping --generations 100 --pop 12 --seeds 1 2 3 4 5

# Legacy trading laboratory
./.venv/bin/python alpha_evolver.py --domain trading --data synthetic --generations 20 --pop 56

# Memory self test
./.venv/bin/python alpha_evolver.py --selftest

# Legacy trading memory self test
./.venv/bin/python alpha_evolver.py --domain trading --selftest

# Sleep phase demo
./.venv/bin/python alpha_evolver.py --domain trading --generations 13 --pop 56 --sleep-now

# Claude mode, requires anthropic and ANTHROPIC_API_KEY in .env
./.venv/bin/python alpha_evolver.py --domain trading --mode claude --generations 4 --pop 24

# Real market data, requires yfinance
./.venv/bin/pip install yfinance
./.venv/bin/python alpha_evolver.py --domain trading --data yfinance --mode offline --generations 8

# Weave traces, requires weave
./.venv/bin/pip install weave
./.venv/bin/python alpha_evolver.py --domain trading --weave --mode claude --generations 6

# Live dashboard, requires streamlit
./.venv/bin/pip install streamlit
./.venv/bin/streamlit run dashboard.py
```

## Architecture

```text
domain adapter: Max Cut or trading
          |
          v
 sandboxed candidate program
          |
          v
 independent bounded verifier
          |
          v
 contextual evidence memory
          |
          v
 learned utility controller
          |
          v
 dream / focus / challenge / recover / sleep
```

## Notes

Max Cut search uses train graph quality only for primary fitness. Development
graphs provide trust and stress verification. Sealed graph families are
evaluated only after search and never enter memory or proposer context.

Before each development verification, contextual memory predicts the candidate's
unseen quality from its train result and the generalization gaps of similar prior
programs. Prediction error is therefore a generalization surprise, not a second
fitness score.

Current evidence is intentionally conservative. Rule based stopping retained
sealed quality while using 45 percent fewer generations than a fixed budget.
Learned stopping retained quality but saved only 13 percent. Warm contextual
memory changed every replay search, but did not improve mean replay quality
across five matched seeds. It remains experimental.

The trading adapter still uses out of sample Sharpe as its headline metric and
preserves its no lookahead rules.

`.env` is ignored by git.

## Credits

Conceptual lineage: FunSearch and AlphaEvolve. This is a research prototype, not
a production optimizer.
