# RightRoute

**Predicting optimal LLM routing from task and prompt features.**

RightRoute learns when a task actually *needs* a stronger (more expensive) model versus when a weaker (cheaper) one is sufficient — enabling cost-efficient, quality-aware model routing.

## Project Overview

This is a KDD (Knowledge Discovery in Databases) pipeline built on two large datasets:

- **LMArena Human Preference 140K** — real human battle outcomes between frontier LLMs
- **WildChat-1M-Tagged** — 837K user prompts with Mistral-assigned category and complexity labels

The core idea: compute Elo ratings from battle data to label which model was "stronger" in each matchup, then train a classifier to predict — from the prompt alone — whether the stronger model would have won.

## Phase 1: Data Acquisition, Target Construction & Feature Enrichment

### Step 1 — Load & Explore Arena Data
- Loads the 135K-row Arena dataset
- Audits winner distribution, model coverage, language breakdown, and category tag frequencies

### Step 2 — Construct the Target Variable
- Computes Elo ratings for all 53 models using a multi-round Bradley-Terry approach (mirrors the official Chatbot Arena methodology)
- Labels each battle: `strong_model_won = 1` if the higher-Elo model won, `0` if the weaker model was good enough
- Filters to battles with Elo gap ≥ 50 to remove uninformative close matchups (~71K battles retained)

### Step 3 — Train a WildChat Prompt Classifier
- Trains both a **TF-IDF + Logistic Regression** baseline and a **DistilBERT** classifier on WildChat-1M-Tagged
- Predicts `category` (task type) and `complexity` (low/medium/high) for every Arena prompt
- Validates predictions against existing Arena category tags
- Analyzes strong-model win rates by predicted category and complexity

### Output
`arena_enriched_phase1.parquet` — the Arena dataset enriched with:
- Elo ratings and gap for both models
- `strong_model_won` target variable
- `predicted_category` and `predicted_complexity` from the WildChat classifier

## Phase 2 (Upcoming)
- Tier 2 feature extraction (constraint count, readability scores, keyword signals)
- Sentence-transformer embeddings
- XGBoost and DistilBERT routing classifiers
- Cost-performance curve evaluation

## Environment

Google Colab with GPU recommended for the DistilBERT training step.

```bash
pip install datasets transformers accelerate scikit-learn pandas numpy matplotlib seaborn textstat
```

## File Structure

```
RightRoute/
├── KDD_Phase1_Pipeline.ipynb   # Main pipeline notebook
└── README.md
```
