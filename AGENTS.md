# Alpha Evolver

A self improving research agent. It proposes trading signal formulas, backtests each against a verifier, stores results in a reinforced memory, and breeds better ones, so its out of sample skill climbs. It predicts before it tests and learns from surprise, distills durable principles during a sleep phase, builds a library of its own motifs, rewrites its own strategy each generation, and rewards novelty and risk that pays off.

## Hard rules
- Pure numpy plus standard library for the core. Guarded optional imports for anthropic, weave, matplotlib, yfinance, so the offline path runs with only numpy.
- No lookahead: weights formed on day t earn day t+1's return.
- Headline metric is out of sample Sharpe (last 30 percent), never in sample. The search only optimizes on the first 70 percent.
- Never fabricate results or recalled facts. The model never recalls from memory; it is handed verbatim records from the store. Every idea is re verified by the backtest.
- Alphas are a sandboxed S expression DSL. Validate before evaluating.

## The memory store is the spine
Every item: key (canonical formula string), expr, stats (is_sharpe, oos_sharpe, turnover, size, ic), predicted_sharpe, prediction_error, strength, uses, created_gen, last_used_gen, novelty, motifs, lesson.
Slow store: principles (text plus the keys that support each), motif library (fragment, strength, uses, avg fitness of containers), strategy (current proposer strategy text plus the hit rate it produced).
Strength update each generation: s = s * decay + (a*norm_fitness + b*used_this_gen + c*norm_surprise). decay 0.95, a 0.5, b 0.3, c 0.2. Evict only the lowest strength when over capacity. norm_surprise = clamp(|prediction_error|/scale, 0, 1).
Retrieval (assemble_context, deterministic, every generation): top 8 by strength, top 3 by novelty, top 3 by surprise, all principles, top 6 motifs, returned verbatim with stats.

## The DSL
S expressions, e.g. ["neg", ["ts_mean", "returns", 2]]. Fields: open, high, low, close, volume, returns. Unary: neg, abs, sign, log1p. Binary: add, sub, mul, div, min, max. Cross sectional (per day): cs_rank, cs_demean, cs_zscore. Time series (causal window): ts_delta, ts_mean, ts_std, ts_zscore, ts_sum, ts_min, ts_max, ts_rank. Windows: 2, 3, 5, 10, 20.

## My style
Terse and direct. No em dashes. No hyphenated compound words in prose or comments. One module at a time, run it and show me the output before moving on.
