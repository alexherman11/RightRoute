# KDD Phase 2 — Development Plan
## Feature Engineering, Model Improvement & Deeper Insight

**Project:** Predicting When Stronger Models Win from Prompt Features
**Dataset:** LMArena Human Preference 140K (79,437 rows after Phase 1 filtering)
**Authors:** Alex Herman
**Phase 1 Deliverable:** `arena_enriched_phase1.parquet` — 79,437 rows × 48 columns with domain-specific Elo ratings, target variable (`strong_model_won`), and native category features
**Phase 1 Classifiers:** Domain-bucket baseline (56.0% acc, 0.36 F1), TF-IDF+LR (54.4% acc, 0.50 F1), DistilBERT (56.4% acc, 0.42 F1)

---

## Phase 1 Recap & Diagnosis

Phase 1 established that model rankings genuinely shift across domains (Kendall's τ = 0.55–0.69 between domain pairs) and that domain-specific Elo is a better target than global Elo (11.8% disagreement rate, domain Elo wins on the disagreement subset). The Elo gap filter at ≥50 successfully removed noise, producing a 56/44 class split.

However, all three classifiers barely exceeded the base rate. The root causes are:

1. **Weak feature representation.** TF-IDF captures lexical signal but misses structural properties of prompts (complexity, constraint density, formality). The native category booleans are coarse — `cat_coding` tells you the domain but nothing about *how hard* the coding problem is.
2. **Inherent label noise.** Ties account for ~16% of battles. Human preference is subjective. Even with a perfect feature set, the ceiling is well below 100%.
3. **Missing feature tiers.** Readability, constraint structure, embedding semantics, and response-side divergence features were all deferred to Phase 2.

Phase 2 attacks all three: richer features to push the decision boundary, a tree ensemble to combine complementary signals, and a proper evaluation framework that measures routing value (not just accuracy).

---

## Notebook Structure

Split Phase 2 into two notebooks to keep each runnable in a single Colab session:

| Notebook | Purpose | GPU Required |
|----------|---------|:---:|
| `Phase2A_Features.ipynb` | Data validation, feature engineering (Tier 2 + Tier 3), EDA, save enriched parquet | Yes (for embeddings) |
| `Phase2B_Models.ipynb` | Ceiling analysis, XGBoost, SHAP, evaluation framework, cost curves | No (CPU is fine) |

Phase2A loads `arena_enriched_phase1.parquet` and outputs `arena_enriched_phase2.parquet` plus `phase2_artifacts.pkl` (PCA model, split indices, feature lists). Phase2B consumes those and produces all final models and results.

---

## Step 2A-0: Data Loading & Validation

Start Phase2A by loading the Phase 1 output and verifying it matches expectations. This catches silent data corruption before spending time on feature engineering.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 50)
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

df = pd.read_parquet('arena_enriched_phase1.parquet')
print(f"Loaded: {df.shape}")
```

### Validation checks

```python
# --- Row count ---
assert len(df) == 79437, f"Expected 79,437 rows, got {len(df)}"

# --- Required columns from Phase 1 ---
required_cols = [
    'prompt_text', 'strong_model_won', 'domain_bucket', 'language',
    'elo_domain_gap', 'stronger_model_name', 'weaker_model_name',
    'prompt_word_count', 'is_code', 'cat_if_score',
    'cat_math', 'cat_coding', 'cat_creative_writing', 'cat_instruction_following',
    'cat_complexity', 'cat_creativity', 'cat_domain_knowledge',
    'cat_problem_solving', 'cat_real_world', 'cat_specificity', 'cat_technical_accuracy',
    'meta_user_tokens', 'meta_turns',
    'meta_assistant_a_tokens', 'meta_assistant_b_tokens',
    'meta_headers_a', 'meta_headers_b',
    'meta_lists_ordered_a', 'meta_lists_unordered_a',
    'meta_lists_ordered_b', 'meta_lists_unordered_b',
    'meta_bold_a', 'meta_bold_b',
]
missing = [c for c in required_cols if c not in df.columns]
assert not missing, f"Missing columns: {missing}"

# --- Class balance ---
pos_rate = df['strong_model_won'].mean()
print(f"Class balance: {pos_rate:.3f} positive (strong won) / {1 - pos_rate:.3f} negative")
assert 0.54 < pos_rate < 0.58, f"Unexpected class balance: {pos_rate:.3f}"

# --- Domain bucket distribution ---
print(f"\nDomain buckets:\n{df['domain_bucket'].value_counts().to_string()}")

# --- Null check on critical columns ---
null_counts = df[required_cols].isnull().sum()
has_nulls = null_counts[null_counts > 0]
if len(has_nulls) > 0:
    print(f"\nWarning — columns with nulls:\n{has_nulls.to_string()}")
else:
    print("\nNo nulls in required columns.")

print("\nValidation passed.")
```

---

## Step 2A-1: Tier 2 Feature Engineering (Lightweight, No GPU)

These features extract structural and linguistic properties of the prompt using only standard Python libraries. They should be computed for every row in `df`.

### 2A-1a: Readability Metrics

Use the `textstat` library (already installed in Phase 1).

```python
import textstat

df['flesch_kincaid_grade'] = df['prompt_text'].apply(textstat.flesch_kincaid_grade)
df['coleman_liau_index'] = df['prompt_text'].apply(textstat.coleman_liau_index)
df['automated_readability'] = df['prompt_text'].apply(textstat.automated_readability_index)
df['reading_ease'] = df['prompt_text'].apply(textstat.flesch_reading_ease)
```

**Rationale:** Higher-grade prompts are more sophisticated and may favor stronger models that handle complex instructions better. Reading ease inversely captures this.

**Edge case:** Very short prompts (≤3 words) produce degenerate readability scores. Clip or impute these to the domain-bucket median.

```python
short_mask = df['prompt_word_count'] <= 3
for col in ['flesch_kincaid_grade', 'coleman_liau_index', 'automated_readability', 'reading_ease']:
    df.loc[short_mask, col] = df.loc[short_mask, 'domain_bucket'].map(
        df.loc[~short_mask].groupby('domain_bucket')[col].median()
    )
```

### 2A-1b: Constraint & Structure Counting

Count explicit instructional constraints via regex patterns.

```python
import re

def count_constraints(text):
    """Count explicit instructional constraints in a prompt."""
    if not text or not isinstance(text, str):
        return {}

    text_lower = text.lower()

    return {
        'n_questions': text.count('?'),
        'n_code_fences': text.count('```'),
        'n_urls': len(re.findall(r'https?://\S+', text)),
        'has_numbered_list': int(bool(re.search(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))),
        'n_negation_constraints': len(re.findall(
            r'\b(do not|don\'t|must not|never|avoid|without|no +\w+ing)\b', text_lower
        )),
        'n_positive_constraints': len(re.findall(
            r'\b(must include|make sure|ensure|always|exactly \d+|in \d+ words|format as|write in)\b', text_lower
        )),
        'n_sub_tasks': len(re.findall(r'^\s*[-•*]\s', text, re.MULTILINE)),
        'has_persona_instruction': int(bool(re.findall(
            r'\b(act as|you are|pretend|role[- ]?play|imagine you)\b', text_lower
        ))),
        'has_output_format': int(bool(re.findall(
            r'\b(json|csv|table|bullet|markdown|xml|yaml|list format)\b', text_lower
        ))),
    }

constraint_df = df['prompt_text'].apply(count_constraints).apply(pd.Series)
df = pd.concat([df, constraint_df], axis=1)

# Derived: total constraint count
df['total_constraints'] = (
    df['n_negation_constraints'] + df['n_positive_constraints'] +
    df['has_numbered_list'] + df['n_sub_tasks'] + df['has_output_format']
)
```

**Rationale:** Heavily constrained prompts are harder to satisfy. Stronger models should have an edge on these. The `total_constraints` aggregate is the most likely single predictor to beat the base rate.

### 2A-1c: Prompt Entropy

Character-level Shannon entropy as a proxy for lexical diversity.

```python
from collections import Counter
import math

def char_entropy(text):
    """Character-level Shannon entropy."""
    if not text or len(text) < 2:
        return 0.0
    counts = Counter(text.lower())
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

df['char_entropy'] = df['prompt_text'].apply(char_entropy)
```

Also compute **word-level entropy** for comparison:

```python
def word_entropy(text):
    """Word-level Shannon entropy."""
    words = text.lower().split() if text else []
    if len(words) < 2:
        return 0.0
    counts = Counter(words)
    n = len(words)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

df['word_entropy'] = df['prompt_text'].apply(word_entropy)
```

### 2A-1d: Relative Prompt Length

Normalize prompt length relative to the domain-bucket median — a 50-word prompt is long for chat but short for coding.

```python
domain_median_length = df.groupby('domain_bucket')['prompt_word_count'].transform('median')
df['relative_prompt_length'] = df['prompt_word_count'] / domain_median_length.clip(lower=1)
df['log_prompt_length'] = np.log1p(df['prompt_word_count'])
```

### 2A-1e: Use `cat_if_score` as a Feature

This was extracted in Phase 1 but never used in any classifier. It's a 0–5 instruction-following difficulty score from the Arena's own annotation pipeline.

```python
# Already in the dataframe — just make sure it's in the feature list and handle NaN
assert 'cat_if_score' in df.columns
# cat_if_score may have NaN values — impute with median
if df['cat_if_score'].isna().sum() > 0:
    df['cat_if_score'] = df['cat_if_score'].fillna(df['cat_if_score'].median())
    print(f"Imputed {df['cat_if_score'].isna().sum()} NaN values in cat_if_score")
```

### 2A-1f: Language & Multi-Script Detection

```python
def detect_non_ascii_ratio(text):
    """Fraction of characters that are non-ASCII (proxy for non-English content)."""
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)

df['non_ascii_ratio'] = df['prompt_text'].apply(detect_non_ascii_ratio)
df['is_english'] = (df['language'] == 'en').astype(int)
```

### Checkpoint 2A-1

After completing all Tier 2 features, verify:

```python
tier2_features = [
    'flesch_kincaid_grade', 'coleman_liau_index', 'automated_readability', 'reading_ease',
    'n_questions', 'n_code_fences', 'n_urls', 'has_numbered_list',
    'n_negation_constraints', 'n_positive_constraints', 'n_sub_tasks',
    'has_persona_instruction', 'has_output_format', 'total_constraints',
    'char_entropy', 'word_entropy',
    'relative_prompt_length', 'log_prompt_length',
    'non_ascii_ratio', 'is_english',
]

# All features should exist and have no NaN after imputation
for f in tier2_features:
    assert f in df.columns, f"Missing feature: {f}"
    assert df[f].isna().sum() == 0, f"NaN values in: {f}"

# Quick sanity: correlations with target
print("--- Tier 2 feature correlations with strong_model_won ---")
corrs = df[tier2_features + ['strong_model_won']].corr()['strong_model_won'].drop('strong_model_won')
print(corrs.sort_values(ascending=False).to_string())
```

**Expected:** Most correlations will be weak (|r| < 0.05). That's fine — individual features don't need to be strong if they're complementary in a tree ensemble. The sanity check is that none are *exactly zero* (which would indicate a data bug) and that the signs make sense (e.g., `total_constraints` should be slightly positive).

---

## Step 2A-2: Tier 3 Feature Engineering (GPU Required)

### 2A-2a: Sentence Transformer Embeddings

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Load model — this is small and fast
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

# Encode all prompts (batched for speed)
embeddings = embedder.encode(
    df['prompt_text'].fillna('').tolist(),
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True
)
# embeddings.shape = (79437, 384)
```

**Dimensionality reduction via PCA:**

```python
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# CRITICAL: Reproduce the SAME train/val/test split from Phase 1.
# Phase 1 split df_filtered (a DataFrame) with these exact parameters.
# Here we split indices into df (which IS df_filtered, loaded from parquet).
# The random_state and stratify ensure identical splits as long as
# the row order in the parquet matches Phase 1's df_filtered.
# -------------------------------------------------------
train_idx, temp_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42,
    stratify=df['strong_model_won']
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42,
    stratify=df.iloc[temp_idx]['strong_model_won']
)

# Verify split sizes match Phase 1
assert len(train_idx) == 63549, f"Train size mismatch: {len(train_idx)} != 63549"
assert len(val_idx) == 7944, f"Val size mismatch: {len(val_idx)} != 7944"
assert len(test_idx) == 7944, f"Test size mismatch: {len(test_idx)} != 7944"
print(f"Split sizes — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
print(f"Train positive rate: {df.iloc[train_idx]['strong_model_won'].mean():.3f}")
print(f"Test positive rate:  {df.iloc[test_idx]['strong_model_won'].mean():.3f}")

# IMPORTANT: fit PCA on train set only to prevent data leakage
pca = PCA(n_components=48, random_state=42)
pca.fit(embeddings[train_idx])

embeddings_reduced = pca.transform(embeddings)
print(f"Explained variance (48 components): {pca.explained_variance_ratio_.sum():.3f}")

# Add as columns
for i in range(48):
    df[f'emb_{i:02d}'] = embeddings_reduced[:, i]
```

**Why 48 dimensions?** This is a tunable hyperparameter. Start with 48 (keeps ~85–90% of variance for MiniLM), and the XGBoost feature importance will tell you if more or fewer are needed. If the top PCA components dominate SHAP importance, try 64. If they're all low, try 32.

### 2A-2b: Response Divergence Features

These use metadata already in the dataframe. They capture how differently the two models responded — high divergence suggests the battle was more decisive.

```python
# Token count divergence
df['response_length_ratio'] = (
    df['meta_assistant_a_tokens'] / df['meta_assistant_b_tokens'].clip(lower=1)
)
df['response_length_diff'] = (
    df['meta_assistant_a_tokens'] - df['meta_assistant_b_tokens']
).abs()

# Formatting divergence
df['header_diff'] = (df['meta_headers_a'] - df['meta_headers_b']).abs()
df['list_diff'] = (
    (df['meta_lists_ordered_a'] + df['meta_lists_unordered_a']) -
    (df['meta_lists_ordered_b'] + df['meta_lists_unordered_b'])
).abs()
df['bold_diff'] = (df['meta_bold_a'] - df['meta_bold_b']).abs()

# Total formatting divergence (composite)
df['formatting_divergence'] = df['header_diff'] + df['list_diff'] + df['bold_diff']
```

**Important caveat for the writeup:** Response divergence features use information from *both models' responses*, which means they can't be used at inference time in a real-time router (you don't have the responses yet when you're deciding where to route). Document this clearly. These features serve the *insight* goal (understanding what drives outcomes) and can be included in an "oracle" or "analysis" model variant but excluded from the "deployable router" variant.

### Checkpoint 2A-2

```python
tier3_features = [f'emb_{i:02d}' for i in range(48)] + [
    'response_length_ratio', 'response_length_diff',
    'header_diff', 'list_diff', 'bold_diff', 'formatting_divergence',
]

for f in tier3_features:
    assert f in df.columns, f"Missing: {f}"
    assert df[f].isna().sum() == 0, f"NaN in: {f}"

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
print(f"Total feature count: {len(tier2_features) + len(tier3_features)}")
```

### 2A-3: Save Enriched Dataset

```python
# Save everything — Phase2B will load this
df.to_parquet('arena_enriched_phase2.parquet', index=False)
print(f"Saved: arena_enriched_phase2.parquet ({df.shape})")

# Also save the PCA model and train/val/test indices for reproducibility
import pickle
with open('phase2_artifacts.pkl', 'wb') as f:
    pickle.dump({
        'pca': pca,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'tier2_features': tier2_features,
        'tier3_features': tier3_features,
    }, f)
```

---

## Step 2A-4: Exploratory Data Analysis

Run this section in Phase2A after all features are computed. The goal is to understand the feature landscape before training models.

### 2A-4a: Feature Distribution Overview

```python
# Histograms of all Tier 2 features, split by target
fig, axes = plt.subplots(5, 4, figsize=(20, 20))
for ax, feat in zip(axes.ravel(), tier2_features):
    for label, color in [(1, 'steelblue'), (0, 'coral')]:
        subset = df[df['strong_model_won'] == label][feat]
        ax.hist(subset, bins=40, alpha=0.5, color=color,
                label='strong_won' if label else 'strong_lost', density=True)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('tier2_feature_distributions.png', dpi=150)
plt.show()
```

### 2A-4b: Feature Correlation Heatmap

Check for redundancy among Tier 2 features before handing them to a model. Highly correlated features (|r| > 0.9) add noise without information.

```python
corr_matrix = df[tier2_features].corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Tier 2 Feature Correlations')
plt.tight_layout()
plt.savefig('tier2_feature_correlations.png', dpi=150)
plt.show()

# Flag highly correlated pairs
high_corr = []
for i in range(len(tier2_features)):
    for j in range(i + 1, len(tier2_features)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.9:
            high_corr.append((tier2_features[i], tier2_features[j], r))
if high_corr:
    print("Highly correlated feature pairs (|r| > 0.9):")
    for f1, f2, r in high_corr:
        print(f"  {f1} <-> {f2}: r={r:.3f}")
    print("Consider dropping one from each pair if XGBoost importance is low for both.")
else:
    print("No feature pairs with |r| > 0.9.")
```

### 2A-4c: Model-Pair Analysis

```python
# Group by (stronger_model, weaker_model) and compute pair-specific win rates
pair_stats = df.groupby(['stronger_model_name', 'weaker_model_name']).agg(
    n_battles=('strong_model_won', 'count'),
    win_rate=('strong_model_won', 'mean'),
).reset_index()

# Filter to pairs with enough data
pair_stats_filtered = pair_stats[pair_stats['n_battles'] >= 30].copy()
pair_stats_filtered = pair_stats_filtered.sort_values('win_rate', ascending=False)

print(f"Model pairs with >=30 battles: {len(pair_stats_filtered)}")
print(f"\n--- Most predictable pairs (strong model dominates) ---")
print(pair_stats_filtered.head(15).to_string(index=False))
print(f"\n--- Least predictable pairs (near coin-flip) ---")
print(pair_stats_filtered.sort_values('win_rate', key=lambda x: (x - 0.5).abs()).head(15).to_string(index=False))
```

Visualize as a heatmap of the top 15 most frequent models:

```python
top_models = df['stronger_model_name'].value_counts().head(15).index.tolist()
pivot = pair_stats_filtered[
    pair_stats_filtered['stronger_model_name'].isin(top_models) &
    pair_stats_filtered['weaker_model_name'].isin(top_models)
].pivot_table(index='stronger_model_name', columns='weaker_model_name', values='win_rate')

plt.figure(figsize=(14, 10))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5, vmin=0.3, vmax=0.7)
plt.title('Strong Model Win Rate by Model Pair')
plt.tight_layout()
plt.savefig('model_pair_heatmap.png', dpi=150)
plt.show()
```

### 2A-4d: Language Stratification

```python
# Win rate by language (top 10 languages only)
top_langs = df['language'].value_counts().head(10).index
lang_stats = df[df['language'].isin(top_langs)].groupby('language').agg(
    n=('strong_model_won', 'count'),
    win_rate=('strong_model_won', 'mean'),
).sort_values('win_rate', ascending=False)

print("--- Strong model win rate by language ---")
print(lang_stats.to_string())

# Interaction: language x domain
lang_domain = df[df['language'].isin(top_langs)].groupby(
    ['language', 'domain_bucket']
)['strong_model_won'].agg(['mean', 'count']).reset_index()
lang_domain = lang_domain[lang_domain['count'] >= 50]

pivot_ld = lang_domain.pivot_table(index='language', columns='domain_bucket', values='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_ld, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5)
plt.title('Win Rate: Language x Domain')
plt.tight_layout()
plt.savefig('language_domain_interaction.png', dpi=150)
plt.show()
```

### Checkpoint 2A-4

Manually verify:
- [ ] Tier 2 feature distributions show some separation between classes (even if slight)
- [ ] Correlation heatmap identifies any redundant feature pairs
- [ ] Model-pair heatmap reveals both predictable (>0.6) and chaotic (~0.5) pairings
- [ ] Language analysis confirms English isn't the only interesting case
- [ ] All plots saved as PNGs for the final report

---

## Step 2B-0: Load Data & Retrain Phase 1 Baselines

Phase2B must be self-contained. Load Phase 2A outputs and retrain the TF-IDF+LR baseline so its predictions are available for comparison and stacking. This also ensures we evaluate all models on the exact same test split.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

# Load Phase 2 enriched data
df = pd.read_parquet('arena_enriched_phase2.parquet')
with open('phase2_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

train_idx = artifacts['train_idx']
val_idx = artifacts['val_idx']
test_idx = artifacts['test_idx']
tier2_features = artifacts['tier2_features']
tier3_features = artifacts['tier3_features']

y_train = df.iloc[train_idx]['strong_model_won'].values
y_val = df.iloc[val_idx]['strong_model_won'].values
y_test = df.iloc[test_idx]['strong_model_won'].values

print(f"Loaded: {df.shape}")
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
print(f"Test positive rate: {y_test.mean():.3f}")
```

### 2B-0a: Retrain TF-IDF + LR (Phase 1 Baseline, for Fair Comparison)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# Exact same configuration as Phase 1
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), max_df=0.95, min_df=5)
X_train_tfidf = tfidf.fit_transform(df.iloc[train_idx]['prompt_text'].fillna(''))
X_val_tfidf = tfidf.transform(df.iloc[val_idx]['prompt_text'].fillna(''))
X_test_tfidf = tfidf.transform(df.iloc[test_idx]['prompt_text'].fillna(''))

cat_feature_cols = ['cat_math', 'cat_coding', 'cat_creative_writing', 'cat_instruction_following']

X_train_combined = hstack([X_train_tfidf, df.iloc[train_idx][cat_feature_cols].values])
X_val_combined = hstack([X_val_tfidf, df.iloc[val_idx][cat_feature_cols].values])
X_test_combined = hstack([X_test_tfidf, df.iloc[test_idx][cat_feature_cols].values])

lr_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
lr_model.fit(X_train_combined, y_train)

lr_probs_test = lr_model.predict_proba(X_test_combined)[:, 1]
lr_preds_test = (lr_probs_test > 0.5).astype(int)
lr_acc = accuracy_score(y_test, lr_preds_test)
lr_f1 = f1_score(y_test, lr_preds_test, average='macro')
lr_auc = roc_auc_score(y_test, lr_probs_test)

print(f"TF-IDF + LR (retrained): Acc={lr_acc:.4f}  F1={lr_f1:.4f}  AUC={lr_auc:.4f}")
```

### 2B-0b: Compute Phase 1 Baselines

```python
# Base rate
base_rate = y_test.mean()
print(f"Base rate (always predict 1): {base_rate:.4f}")

# Domain-bucket baseline: per-domain win rate from training set
domain_win_rates = df.iloc[train_idx].groupby('domain_bucket')['strong_model_won'].mean()
baseline_preds = df.iloc[test_idx]['domain_bucket'].map(domain_win_rates).round().astype(int)
baseline_acc = accuracy_score(y_test, baseline_preds)
baseline_f1 = f1_score(y_test, baseline_preds, average='macro')
print(f"Domain-Bucket Baseline: Acc={baseline_acc:.4f}  F1={baseline_f1:.4f}")

# Phase 1 DistilBERT result (hardcoded — model weights exist but retraining
# is expensive and not the focus of Phase 2)
bert_acc = 0.5641
bert_f1 = 0.4189
print(f"DistilBERT (Phase 1, reported): Acc={bert_acc:.4f}  F1={bert_f1:.4f}")
```

---

## Step 2B-1: Ceiling Analysis

Before training new models, establish the theoretical performance ceiling. This sets expectations and prevents chasing impossible gains.

### 2B-1a: Oracle Baseline (Elo Gap Threshold)

If you knew the Elo gap and used a simple threshold, how well could you do?

```python
# Oracle: use Elo gap as a score — higher gap = more confident strong wins
# This is the best you could do if Elo gap were the only feature
elo_gaps = df.iloc[test_idx]['elo_domain_gap'].values
oracle_auc = roc_auc_score(y_test, elo_gaps)
print(f"Oracle AUC (Elo gap as score): {oracle_auc:.4f}")
```

### 2B-1b: Noise Floor from Tie Rate

```python
# Ties in the original data represent inherent label noise.
# (We filtered both_bad, but ties are counted as 0.5 in Elo, and in our target
# they go to strong_model_won=0 since tie != strong model winning.)
# The tie rate gives a rough bound on irreducible error.
tie_rate_original = 0.159  # from Phase 1 audit
print(f"Original tie rate: {tie_rate_original:.1%}")
print(f"Rough noise floor: a perfect model would still get ~{tie_rate_original/2:.1%} of battles 'wrong'")
print(f"Theoretical accuracy ceiling: ~{1 - tie_rate_original/2:.1%}")
```

---

## Step 2B-2: XGBoost Routing Model

### 2B-2a: Feature Matrix Assembly

```python
import xgboost as xgb

# Define feature sets for two model variants
tier1_features = [
    'prompt_word_count', 'cat_if_score',
    'cat_math', 'cat_coding', 'cat_creative_writing', 'cat_instruction_following',
    'cat_complexity', 'cat_creativity', 'cat_domain_knowledge',
    'cat_problem_solving', 'cat_real_world', 'cat_specificity', 'cat_technical_accuracy',
    'is_code', 'meta_user_tokens', 'meta_turns',
]

tier3_embedding_features = [f'emb_{i:02d}' for i in range(48)]

# Response-side features (analysis model only — NOT usable for live routing)
response_features = [
    'response_length_ratio', 'response_length_diff',
    'header_diff', 'list_diff', 'bold_diff', 'formatting_divergence',
]

# Domain bucket as one-hot columns
domain_dummies = pd.get_dummies(df['domain_bucket'], prefix='domain')
domain_dummy_cols = domain_dummies.columns.tolist()
for col in domain_dummy_cols:
    df[col] = domain_dummies[col]

# ROUTER MODEL: prompt-only features (deployable)
router_features = tier1_features + tier2_features + tier3_embedding_features + domain_dummy_cols

# ANALYSIS MODEL: includes response features (for insight, not deployment)
analysis_features = router_features + response_features

print(f"Router features: {len(router_features)}")
print(f"Analysis features: {len(analysis_features)}")
```

### 2B-2b: Train XGBoost (Router Variant)

```python
X_train = df.iloc[train_idx][router_features].values
X_val = df.iloc[val_idx][router_features].values
X_test = df.iloc[test_idx][router_features].values

# Handle class imbalance (mild: 56/44)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=router_features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=router_features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=router_features)

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'scale_pos_weight': scale_pos_weight,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'seed': 42,
}

model_router = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=50,
)

# Save model
model_router.save_model('xgb_router.json')
```

### 2B-2c: Hyperparameter Tuning (Optuna, 25 Trials)

After getting initial results, tune with Bayesian optimization. This is faster and more effective than exhaustive grid search.

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Combine train + val for CV (keep test held out)
cv_idx = np.concatenate([train_idx, val_idx])
X_cv = df.iloc[cv_idx][router_features].values
y_cv = df.iloc[cv_idx]['strong_model_won'].values
dcv = xgb.DMatrix(X_cv, label=y_cv, feature_names=router_features)

def objective(trial):
    p = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'seed': 42,
    }
    cv_result = xgb.cv(
        p, dcv, num_boost_round=500,
        nfold=5, stratified=True, seed=42,
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return cv_result['test-auc-mean'].max()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)

print(f"Best CV AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

Retrain on full train+val with best params and evaluate on test:

```python
best_params = study.best_params
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'seed': 42,
})

model_router_tuned = xgb.train(
    best_params, dcv,
    num_boost_round=500,
    verbose_eval=False,
)
model_router_tuned.save_model('xgb_router_tuned.json')

# Use the tuned model for all downstream evaluation
xgb_probs = model_router_tuned.predict(dtest)
xgb_preds = (xgb_probs > 0.5).astype(int)
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds, average='macro')
xgb_auc = roc_auc_score(y_test, xgb_probs)

print(f"\nTuned XGBoost Router: Acc={xgb_acc:.4f}  F1={xgb_f1:.4f}  AUC={xgb_auc:.4f}")
```

### 2B-2d: Train Analysis Model (with response features)

Same tuned params but using `analysis_features` instead of `router_features`. Compare the two models to quantify how much signal lives in the response vs. the prompt.

```python
X_train_analysis = df.iloc[train_idx][analysis_features].values
X_val_analysis = df.iloc[val_idx][analysis_features].values
X_test_analysis = df.iloc[test_idx][analysis_features].values

dtrain_analysis = xgb.DMatrix(X_train_analysis, label=y_train, feature_names=analysis_features)
dval_analysis = xgb.DMatrix(X_val_analysis, label=y_val, feature_names=analysis_features)
dtest_analysis = xgb.DMatrix(X_test_analysis, label=y_test, feature_names=analysis_features)

model_analysis = xgb.train(
    best_params, dtrain_analysis,
    num_boost_round=1000,
    evals=[(dtrain_analysis, 'train'), (dval_analysis, 'val')],
    early_stopping_rounds=50,
    verbose_eval=False,
)
model_analysis.save_model('xgb_analysis.json')

analysis_probs = model_analysis.predict(dtest_analysis)
analysis_preds = (analysis_probs > 0.5).astype(int)
analysis_acc = accuracy_score(y_test, analysis_preds)
analysis_f1 = f1_score(y_test, analysis_preds, average='macro')
analysis_auc = roc_auc_score(y_test, analysis_probs)

print(f"Analysis Model: Acc={analysis_acc:.4f}  F1={analysis_f1:.4f}  AUC={analysis_auc:.4f}")
print(f"Response feature lift: AUC +{analysis_auc - xgb_auc:.4f}")
```

### Checkpoint 2B-2

```python
# Summary of all models so far
print(f"{'Model':<30} {'Acc':>8} {'F1':>8} {'AUC':>8}")
print("-" * 56)
print(f"{'Base Rate (always predict 1)':<30} {base_rate:>8.4f} {'—':>8} {'0.500':>8}")
print(f"{'Domain-Bucket Baseline':<30} {baseline_acc:>8.4f} {baseline_f1:>8.4f} {'—':>8}")
print(f"{'TF-IDF + LR':<30} {lr_acc:>8.4f} {lr_f1:>8.4f} {lr_auc:>8.4f}")
print(f"{'DistilBERT (Phase 1)':<30} {bert_acc:>8.4f} {bert_f1:>8.4f} {'—':>8}")
print(f"{'XGBoost Router (tuned)':<30} {xgb_acc:>8.4f} {xgb_f1:>8.4f} {xgb_auc:>8.4f}")
print(f"{'XGBoost Analysis (oracle)':<30} {analysis_acc:>8.4f} {analysis_f1:>8.4f} {analysis_auc:>8.4f}")
```

**Expected:** Router model AUC should be 0.55–0.62 (modest but above chance). Analysis model AUC should be 2–5 points higher, confirming response divergence carries signal. If the router model doesn't beat TF-IDF+LR (AUC ~0.52), something is wrong with the feature pipeline — debug before proceeding.

---

## Step 2B-3: SHAP Analysis

This is where the KDD-level insight lives. SHAP values tell you *why* the model makes each prediction.

```python
import shap

explainer = shap.TreeExplainer(model_router_tuned)
shap_values = explainer.shap_values(X_test)  # shape: (n_test, n_features)

# Beeswarm plot — top 20 features
shap.summary_plot(shap_values, X_test, feature_names=router_features, max_display=20, show=False)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# Bar plot — mean absolute SHAP
shap.summary_plot(shap_values, X_test, feature_names=router_features,
                  plot_type='bar', max_display=20, show=False)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2B-3a: Dependence Plots for Top Features

```python
# Get top 5 features by mean |SHAP|
top_features_idx = np.argsort(-np.abs(shap_values).mean(axis=0))[:5]

for idx in top_features_idx:
    feat_name = router_features[idx]
    shap.dependence_plot(idx, shap_values, X_test, feature_names=router_features, show=False)
    plt.tight_layout()
    plt.savefig(f'shap_dependence_{feat_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Checkpoint 2B-3

- [ ] Beeswarm plot saved — identifies which features push predictions toward strong_won vs strong_lost
- [ ] Bar plot saved — ranks features by overall importance
- [ ] Dependence plots saved for top 5 features
- [ ] Write a brief narrative interpreting the SHAP results (2–3 paragraphs for the report)

---

## Step 2B-4: Evaluation Framework

### 2B-4a: Confusion Matrices

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = ['strong_lost', 'strong_won']

for ax, (name, preds) in zip(axes, [
    ('TF-IDF + LR', lr_preds_test),
    ('XGBoost Router', xgb_preds),
    ('XGBoost Analysis', analysis_preds),
]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(f'{name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
```

**What to check:** Are models just predicting the majority class? If the "strong_lost" column is nearly empty, the model learned to always predict 1 (i.e., it collapsed to the base rate). The confusion matrix makes this failure mode immediately visible.

### 2B-4b: ROC Curves

```python
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

for name, probs, color, ls in [
    ('TF-IDF + LR', lr_probs_test, 'gray', '--'),
    ('XGBoost Router', xgb_probs, 'steelblue', '-'),
    ('XGBoost Analysis', analysis_probs, 'coral', '-'),
    ('Elo Gap (oracle)', elo_gaps, 'green', ':'),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
            label=f'{name} (AUC={auc_val:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models')
ax.legend(loc='lower right')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()
```

### 2B-4c: Bootstrap Confidence Intervals

With models barely above base rate, prove the improvements are statistically real.

```python
from sklearn.utils import resample

def bootstrap_metric(y_true, y_probs, metric_fn, n_iter=1000, seed=42):
    """Compute bootstrap 95% CI for a metric."""
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_iter):
        idx = rng.randint(0, len(y_true), len(y_true))
        # Skip degenerate resamples (single class)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_probs[idx]))
    scores = np.array(scores)
    return np.median(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

print("--- 95% Bootstrap Confidence Intervals (1000 iterations) ---")
print(f"{'Model':<25} {'AUC':>8}   {'95% CI':>16}")
print("-" * 55)

for name, probs in [
    ('TF-IDF + LR', lr_probs_test),
    ('XGBoost Router', xgb_probs),
    ('XGBoost Analysis', analysis_probs),
]:
    median_auc, lo, hi = bootstrap_metric(y_test, probs, roc_auc_score)
    print(f"{name:<25} {median_auc:>8.4f}   [{lo:.4f}, {hi:.4f}]")
```

**Interpretation:** If the XGBoost Router CI overlaps with the TF-IDF + LR CI, the improvement is not statistically significant and the richer features haven't helped. If the CIs don't overlap, Phase 2 features provide genuine lift.

### 2B-4d: Per-Domain Performance Breakdown

```python
print(f"\n{'Domain':<20} {'Model':<20} {'Acc':>8} {'AUC':>8} {'n':>6}")
print("-" * 65)

for domain in ['code', 'math_science', 'creative_writing', 'chat']:
    mask = df.iloc[test_idx]['domain_bucket'].values == domain
    n = mask.sum()
    yt = y_test[mask]

    for name, probs in [('TF-IDF + LR', lr_probs_test), ('XGBoost Router', xgb_probs)]:
        preds = (probs[mask] > 0.5).astype(int)
        acc = accuracy_score(yt, preds)
        auc = roc_auc_score(yt, probs[mask]) if len(np.unique(yt)) > 1 else float('nan')
        print(f"{domain:<20} {name:<20} {acc:>8.4f} {auc:>8.4f} {n:>6}")
```

### 2B-4e: Confidence Calibration (Reliability Diagrams)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, probs) in zip(axes, [('XGBoost Router', xgb_probs), ('TF-IDF + LR', lr_probs_test)]):
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=15, strategy='uniform')
    ax.plot(prob_pred, prob_true, 'o-', color='steelblue', label=name)
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(f'{name} — Reliability Diagram')
    ax.legend()

plt.tight_layout()
plt.savefig('calibration_plots.png', dpi=150)
plt.show()
```

**Why this matters:** If the XGBoost model says P(strong_wins) = 0.7 but the actual rate is only 0.58, the router's threshold decisions will be miscalibrated. If calibration is poor, apply Platt scaling or isotonic regression as a post-processing step.

### 2B-4f: Cost-Performance Curves

This is the routing-specific evaluation. Define a cost model and sweep the routing threshold.

```python
def compute_routing_curve(y_true, probs, cost_strong=10.0, cost_weak=1.0):
    """
    For each threshold theta, compute:
    - fraction routed to strong model (where P(strong_wins) > theta)
    - average quality (win rate of the routing decision)
    - total cost
    """
    thresholds = np.linspace(0, 1, 101)
    results = []

    for theta in thresholds:
        route_to_strong = probs > theta
        frac_strong = route_to_strong.mean()

        # Quality: for queries routed to strong, did strong actually win?
        # For queries routed to weak, did weak actually win (= strong lost)?
        correct_strong = (y_true[route_to_strong] == 1).sum() if route_to_strong.any() else 0
        correct_weak = (y_true[~route_to_strong] == 0).sum() if (~route_to_strong).any() else 0
        quality = (correct_strong + correct_weak) / len(y_true)

        cost = frac_strong * cost_strong + (1 - frac_strong) * cost_weak

        results.append({
            'threshold': theta,
            'frac_strong': frac_strong,
            'quality': quality,
            'cost': cost,
        })

    return pd.DataFrame(results)

# Compute curves for each model
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, probs, color in [
    ('XGBoost Router', xgb_probs, 'steelblue'),
    ('TF-IDF + LR', lr_probs_test, 'gray'),
]:
    curve = compute_routing_curve(y_test, probs)

    # Quality vs Cost
    axes[0].plot(curve['cost'], curve['quality'], '-', color=color, label=name, linewidth=2)

    # Quality vs Fraction routed to strong
    axes[1].plot(curve['frac_strong'], curve['quality'], '-', color=color, label=name, linewidth=2)

# Baselines
axes[0].axhline(base_rate, color='red', linestyle='--', alpha=0.5, label='Always strong')
axes[0].set_xlabel('Average cost per query')
axes[0].set_ylabel('Routing quality (accuracy)')
axes[0].set_title('Cost-Quality Tradeoff')
axes[0].legend()

axes[1].axhline(base_rate, color='red', linestyle='--', alpha=0.5, label='Always strong')
axes[1].set_xlabel('Fraction routed to strong model')
axes[1].set_ylabel('Routing quality (accuracy)')
axes[1].set_title('Quality vs. Strong Model Usage')
axes[1].legend()

plt.tight_layout()
plt.savefig('cost_performance_curves.png', dpi=150)
plt.show()

# Print recommended operating point
curve = compute_routing_curve(y_test, xgb_probs)
# Find threshold where quality is maximized
best_row = curve.loc[curve['quality'].idxmax()]
print(f"\nRecommended operating point:")
print(f"  Threshold: {best_row['threshold']:.2f}")
print(f"  Quality: {best_row['quality']:.4f}")
print(f"  Fraction routed to strong: {best_row['frac_strong']:.2%}")
print(f"  Cost: {best_row['cost']:.2f} (vs {10.0:.2f} for always-strong)")
```

### 2B-4g: Prompt Difficulty Estimation

Use model confidence to characterize "easy" vs "hard" prompts.

```python
# Bin prompts by XGBoost confidence
df_test = df.iloc[test_idx].copy()
df_test['xgb_prob'] = xgb_probs

# Define difficulty bins
df_test['difficulty'] = pd.cut(
    df_test['xgb_prob'],
    bins=[0, 0.4, 0.6, 1.0],
    labels=['hard_for_strong', 'ambiguous', 'easy_for_strong']
)

print("--- Prompt difficulty distribution ---")
print(df_test['difficulty'].value_counts())

# Characterize each difficulty bin
for diff in ['easy_for_strong', 'ambiguous', 'hard_for_strong']:
    subset = df_test[df_test['difficulty'] == diff]
    print(f"\n--- {diff} (n={len(subset)}) ---")
    print(f"  Actual win rate: {subset['strong_model_won'].mean():.3f}")
    print(f"  Mean word count: {subset['prompt_word_count'].mean():.0f}")
    print(f"  Mean constraints: {subset['total_constraints'].mean():.1f}")
    print(f"  Mean readability: {subset['flesch_kincaid_grade'].mean():.1f}")
    print(f"  Domain dist: {subset['domain_bucket'].value_counts(normalize=True).round(2).to_dict()}")
    print(f"  Code fraction: {subset['is_code'].mean():.2f}")
```

### 2B-4h: Error Analysis

Inspect misclassified examples to understand qualitatively where the router fails.

```python
df_test['xgb_pred'] = xgb_preds
df_test['correct'] = (df_test['xgb_pred'] == df_test['strong_model_won']).astype(int)

# False positives: predicted strong_won=1, actually 0 (routed to strong unnecessarily)
fp = df_test[(df_test['xgb_pred'] == 1) & (df_test['strong_model_won'] == 0)]
# False negatives: predicted strong_won=0, actually 1 (should have routed to strong)
fn = df_test[(df_test['xgb_pred'] == 0) & (df_test['strong_model_won'] == 1)]

print(f"False positives (wasted strong model): {len(fp)} ({len(fp)/len(df_test):.1%})")
print(f"False negatives (needed strong, got weak): {len(fn)} ({len(fn)/len(df_test):.1%})")

print(f"\n--- Sample False Negatives (should have routed to strong) ---")
for _, row in fn.sample(min(10, len(fn)), random_state=42).iterrows():
    prompt_preview = row['prompt_text'][:150].replace('\n', ' ')
    print(f"  [{row['domain_bucket']}] p={row['xgb_prob']:.3f} | {prompt_preview}...")

print(f"\n--- Sample False Positives (wasted strong model) ---")
for _, row in fp.sample(min(10, len(fp)), random_state=42).iterrows():
    prompt_preview = row['prompt_text'][:150].replace('\n', ' ')
    print(f"  [{row['domain_bucket']}] p={row['xgb_prob']:.3f} | {prompt_preview}...")
```

### 2B-4i: Full Comparison Table

```python
results = pd.DataFrame({
    'Model': [
        'Base Rate (always predict 1)',
        'Domain-Bucket Baseline',
        'TF-IDF + LR',
        'DistilBERT (Phase 1)',
        'XGBoost Router (tuned)',
        'XGBoost Analysis (oracle)',
    ],
    'Accuracy': [base_rate, baseline_acc, lr_acc, bert_acc, xgb_acc, analysis_acc],
    'Macro F1': ['—', f'{baseline_f1:.4f}', f'{lr_f1:.4f}', f'{bert_f1:.4f}',
                 f'{xgb_f1:.4f}', f'{analysis_f1:.4f}'],
    'AUC-ROC': ['0.5000', '—', f'{lr_auc:.4f}', '—',
                f'{xgb_auc:.4f}', f'{analysis_auc:.4f}'],
})
print("\n=== Final Model Comparison ===")
print(results.to_string(index=False))
```

---

## Final Checkpoint Checklist

Before considering Phase 2 complete, verify all of the following:

### Artifacts Produced
- [ ] `arena_enriched_phase2.parquet` — full dataset with all Tier 2 + Tier 3 features
- [ ] `phase2_artifacts.pkl` — PCA model, train/val/test indices, feature lists
- [ ] `xgb_router.json` — initial XGBoost router model
- [ ] `xgb_router_tuned.json` — tuned XGBoost router model (best Optuna params)
- [ ] `xgb_analysis.json` — XGBoost analysis model (with response features)

### Plots Saved
- [ ] `tier2_feature_distributions.png`
- [ ] `tier2_feature_correlations.png`
- [ ] `model_pair_heatmap.png`
- [ ] `language_domain_interaction.png`
- [ ] `shap_beeswarm.png`
- [ ] `shap_bar.png`
- [ ] `shap_dependence_*.png` (5 files)
- [ ] `confusion_matrices.png`
- [ ] `roc_curves.png`
- [ ] `calibration_plots.png`
- [ ] `cost_performance_curves.png`

### Results Documented
- [ ] Full comparison table (6 models x 3 metrics)
- [ ] Bootstrap 95% CIs on AUC for key models
- [ ] Per-domain breakdown for TF-IDF+LR and XGBoost
- [ ] Top SHAP features with narrative interpretation
- [ ] Confusion matrices showing prediction distribution
- [ ] Prompt difficulty characterization (easy/ambiguous/hard profiles)
- [ ] Error analysis with sample misclassified prompts
- [ ] Cost-performance analysis with recommended operating point

### Key Questions Answered
- [ ] How much do Tier 2/3 features improve over Phase 1 classifiers? (bootstrap CIs prove it)
- [ ] What fraction of routing signal comes from prompt features vs. response features? (router vs analysis AUC gap)
- [ ] Which prompt properties most strongly predict whether the strong model wins? (SHAP)
- [ ] Does routing behavior differ meaningfully across domains and languages? (per-domain breakdown)
- [ ] What is the achievable cost-quality tradeoff using prompt-only routing? (cost curves + operating point)
- [ ] Where does the router fail? (error analysis)

---

## Appendix: Dependency List

```
# Phase 2A (GPU recommended)
pip install sentence-transformers textstat

# Phase 2B (CPU fine)
pip install xgboost shap optuna
```

All other dependencies (sklearn, pandas, numpy, matplotlib, seaborn) are already installed from Phase 1.

---

## Appendix: Changes from v1 of this Plan

This plan incorporates the following fixes and improvements over the original draft:

1. **Fixed input file reference.** Changed from `arena_processed_step2.parquet` to `arena_enriched_phase1.parquet` (the actual Phase 1 output).
2. **Fixed Phase 1 results.** DistilBERT achieved 56.4% acc / 0.42 F1 (gains over TF-IDF+LR), not "~no gain."
3. **Added data loading validation (Step 2A-0).** Catches data corruption before feature engineering.
4. **Fixed train/val/test split consistency.** Added assertions verifying split sizes match Phase 1 (63,549 / 7,944 / 7,944).
5. **Fixed stacking ensemble dependency bug.** Replaced with self-contained TF-IDF+LR retraining in Phase 2B (Step 2B-0). Removed stacking ensemble (weak base models make it unlikely to help for MVP).
6. **Fixed analysis model code.** Was incomplete with wrong variable names; now fully specified.
7. **Added feature correlation heatmap (Step 2A-4b).** Identifies redundant features before modeling.
8. **Added confusion matrices (Step 2B-4a).** Detects majority-class prediction collapse.
9. **Added ROC curves (Step 2B-4b).** Standard evaluation artifact, was missing.
10. **Added bootstrap confidence intervals (Step 2B-4c).** Critical — proves improvements are statistically real given models are close to base rate.
11. **Added error analysis (Step 2B-4h).** Qualitative insight into failure modes.
12. **Replaced 81-combo grid search with Optuna 25 trials (Step 2B-2c).** Faster, explores continuous space, same or better results.
13. **Removed SHAP interaction effects.** Computationally expensive, not MVP-critical. Beeswarm + dependence plots suffice.
14. **Removed stacking ensemble.** Phase 1 base models are too weak for stacking to meaningfully help. XGBoost Router is the core contribution.
15. **Removed temporal sanity check.** No timestamp column exists in the Phase 1 output; the check would silently no-op.
16. **Added NaN handling for `cat_if_score`.** Phase 1 data may have missing values in this column.
