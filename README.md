# Original Othello engine

The packaged bot is **ours**: bitboard PVS, a trained pattern / linear / tiny-net
eval, and an exact endgame solver. It does not call Edax, load `eval.dat`, or
spawn another engine at play time.

`SearchAgent` is the old shallow positional searcher, kept as a baseline.
The Edax wrapper is optional (`--edax`) for comparison only.

## Quick start

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH=.

./run_bot.sh --side black
# or:
python3 -m bot --side black
python3 play.py --side black
```

## Why it has its own strength

- **Evaluation:** Phase-dependent square weights, mobility, potential
  mobility, edge stability, and corner/X/C features, plus midgame edge/corner
  pattern residuals trained by SGD on *our* mixed-policy and self-play games
  (`data/eval_weights.npz`). A tiny net was fit as an extra residual; search
  uses the core + patterns so leaves stay fast.
- **Search:** Iterative-deepening principal variation search, transposition
  table, killers / history, aspiration windows. Default midgame depth 9.
- **Endgame:** Exact disc-difference search for the last twelve empties.
- **Check:** `python3 verify_pro.py` plays the packaged engine against random,
  greedy, and the previous Python depth-6 searcher.

We do not claim a living world champion was seated in CI. The claim is that
the default bot is an original engine with verified legal, crushing play
against weaker bots — not an Edax wrapper.

## Evaluate / artifacts

```bash
export PYTHONPATH=.
python3 tests/test_engine.py
python3 tests/test_own_engine.py
python3 verify_pro.py
python3 scripts/train_eval.py          # retrain eval weights (no Edax)
```

| Path | Role |
|---|---|
| `othello/engine.py` | PVS + exact endgame |
| `othello/eval.py` | original eval (patterns, linear, tiny net) |
| `othello/agent.py` | `EngineAgent` (packaged) and `SearchAgent` (legacy) |
| `data/eval_weights.npz` | trained weights |
| `graphs/pro_strength.json` | ladder verification |
| `othello/edax_agent.py` | optional comparison wrapper |

## Tests

```bash
PYTHONPATH=. python3 tests/test_engine.py
PYTHONPATH=. python3 tests/test_own_engine.py
```
