# Load Step 2 checkpoint so Step 3 can run independently
import os, pandas as pd

step2_checkpoint = 'arena_processed_step2.parquet'

if os.path.exists(step2_checkpoint):
    df_filtered = pd.read_parquet(step2_checkpoint)
    print(f"Loaded Step 2 checkpoint: {df_filtered.shape}")
else:
    print(f"No checkpoint found at '{step2_checkpoint}' — run Step 2 first, or continue from above.")
