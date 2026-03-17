# RightRoute — Phase 2: Feature Engineering & Routing Classifiers

## Project Context

RightRoute predicts whether a task needs a stronger (expensive) LLM or if a weaker (cheaper) one suffices. Phase 1 built the foundation: Elo ratings from Arena battle data, a `strong_model_won` binary target. Output: `arena_enriched_phase1.parquet` (~71K filtered battles with Elo gap >= 50).

## Phase 2 Goals

1. **Tier 2 Feature Extraction** — Enrich each prompt with constraint count, readability scores (via `textstat`), keyword signals, and prompt length features.
2. **Sentence-Transformer Embeddings** — Generate dense embeddings for Arena prompts (e.g., `all-MiniLM-L6-v2`) as classifier input.
3. **Routing Classifiers** — Train XGBoost (on engineered features) and DistilBERT (on raw text) to predict `strong_model_won`.
4. **Cost-Performance Curve** — Evaluate the routing threshold: at a given confidence cutoff, what % of traffic routes to the cheap model and what accuracy is retained?

## Key Files

- `KDD_Phase1_Pipeline.ipynb` — Phase 1 notebook (Steps 1-3: data loading, Elo/target construction, WildChat classifier)
- `Phase2A_Features.ipynb` — Phase 2A notebook (data validation, Tier 2 feature engineering, EDA)
- `Phase2B_Models.ipynb` — Phase 2B notebook (ceiling analysis, XGBoost routing, SHAP, evaluation, cost curves)
- `arena_enriched_phase1.parquet` — Phase 1 output, input for Phase 2
- `arena_enriched_phase2.parquet` — Phase 2A output (enriched with Tier 2 features + embeddings)
- `best_routing_model.pt` — Saved DistilBERT checkpoint from Phase 1 Step 3
- `xgb_router_tuned.json` — Tuned XGBoost routing model from Phase 2B
- `step3_load_checkpoint.py` — Utility to reload Step 2 intermediate data
- `nb_edit.py` — CLI tool for fine-grained notebook cell editing and execution (see below)

## Tech Stack

- Python, pandas, numpy, scikit-learn, XGBoost
- Hugging Face transformers + accelerate (DistilBERT)
- sentence-transformers (embeddings)
- textstat (readability features)
- matplotlib, seaborn (visualization)
- GPU in use, local CUDA

## Conventions

- Pipeline lives in Jupyter notebooks, one per phase
- Checkpoints saved as `.parquet` between steps for resumability
- Model artifacts saved as `.pt` (PyTorch)
- Target variable: `strong_model_won` (1 = stronger model needed, 0 = weaker sufficient)
- Elo gap >= 50 filter applied in Phase 1; Phase 2 works on the filtered set

## nb_edit.py — Notebook Cell Editor

CLI tool for reading, editing, and running individual notebook cells without opening Jupyter. Useful for targeted fixes and inspecting outputs.

**Commands:**

```bash
# List all cells with type, first line, and output summary
python nb_edit.py <notebook> list

# Read source code of a specific cell (0-indexed)
python nb_edit.py <notebook> read <cell>

# Read the output/results of a cell
python nb_edit.py <notebook> output <cell>

# Replace a cell's source from a file
python nb_edit.py <notebook> replace <cell> <code_file.py>

# Insert a new cell at a position (code or markdown)
python nb_edit.py <notebook> insert <cell> <code_file.py> [--type markdown]

# Delete a cell
python nb_edit.py <notebook> delete <cell>

# Run a single cell (executes all prior cells for context)
python nb_edit.py <notebook> run <cell> [--timeout 120]

# Run a range of cells [start, end] inclusive
python nb_edit.py <notebook> run-range <start> <end> [--timeout 120]
```

**Typical workflow for fixing a cell:**
1. `python nb_edit.py Phase2B_Models.ipynb list` — find the cell number
2. `python nb_edit.py Phase2B_Models.ipynb read 12` — inspect current source
3. `python nb_edit.py Phase2B_Models.ipynb output 12` — check existing output
4. Write fixed code to a temp file, then: `python nb_edit.py Phase2B_Models.ipynb replace 12 fix.py`
5. `python nb_edit.py Phase2B_Models.ipynb run 12` — execute and verify
