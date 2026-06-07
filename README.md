# Alpha Evolver

Alpha Evolver is a small research demo for self improving trading signals. It proposes sandboxed S expression alphas, verifies each one with a no lookahead backtest, stores the verified record in reinforced memory, then breeds the next generation from the strongest, most novel, and most surprising ideas.

The default path is offline and uses only numpy plus the standard library. Optional integrations are guarded for Claude, Weave, yfinance, matplotlib, and Streamlit.

## 3 minute demo

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy matplotlib
./.venv/bin/python alpha_evolver.py --generations 20 --pop 56
```

Watch the per generation log. The synthetic panel has a planted short window reversal and a faint latent volume edge, so the best out of sample Sharpe should usually climb over the run. The command writes:

- `memory.json`
- `history.csv`
- `learning_curve.png`

## Run commands

```bash
# Offline synthetic demo
./.venv/bin/python alpha_evolver.py --mode offline --data synthetic --generations 20 --pop 56

# Memory self test
./.venv/bin/python alpha_evolver.py --selftest

# Sleep phase demo
./.venv/bin/python alpha_evolver.py --generations 13 --pop 56 --sleep-now

# Claude mode, requires anthropic and ANTHROPIC_API_KEY in .env
./.venv/bin/python alpha_evolver.py --mode claude --generations 4 --pop 24

# Real market data, requires yfinance
./.venv/bin/pip install yfinance
./.venv/bin/python alpha_evolver.py --data yfinance --mode offline --generations 8

# Weave traces, requires weave
./.venv/bin/pip install weave
./.venv/bin/python alpha_evolver.py --weave --mode claude --generations 6

# Live dashboard, requires streamlit
./.venv/bin/pip install streamlit
./.venv/bin/streamlit run dashboard.py
```

## Architecture

```text
synthetic or yfinance panel
          |
          v
 sandboxed S expression DSL
          |
          v
 no lookahead verifier
          |
          v
 reinforced memory store
   |      |       |
   |      |       +--> lessons
   |      +----------> motifs
   +-----------------> principles and strategy
          |
          v
 proposer: offline genetic or Claude
          |
          v
 next generation
```

## Notes

The headline metric is out of sample Sharpe on the last 30 percent of the panel. The search objective uses only the first 70 percent. Weights formed on day `t` earn `fwd_ret[t]`, which is the return at day `t + 1`.

`.env` is ignored by git.

## Credits

Conceptual lineage: FunSearch and AlphaEvolve. This repo is a compact educational build, not a production trading system.
