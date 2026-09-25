# Shoe Market — when to stop searching and commit?

Interactive Streamlit lab for **competitive optimal stopping**, built on a
numpy-only simulation engine.

- **Live lab**: set shoppers, styles, time, stock, taste agreement and
  satisfaction distributions, then train agents by self-play evolution and
  watch the learned threshold strategy (browse time, starting pickiness,
  closing-time bar) vs fixed heuristics (37% rule, fixed 0.7, random).
- **Sweep findings**: charts from the full 27-configuration study
  (`sweep_results.csv`).
- **How it works**: the model in plain words.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — the Streamlit app
- `engine.py` — the market simulation (numpy only; `import engine` works standalone)
- `sweep_results.csv` — precomputed results of the full study (27 configs × baselines)
- `../shoe-market-paper.pdf` — the findings paper (in `~/workspace/your_files/`)

The engine's full experiment code, training scripts and detailed findings live
in `~/workspace/shoe-market/` (`engine.py`, `train.py`, `run_experiments.py`,
`FINDINGS.md`).
