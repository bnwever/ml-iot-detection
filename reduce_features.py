import pandas as pd
import numpy as np

def reduce_features(filepath):
    print(f"Loading data from {filepath}...")
    X = pd.read_csv(filepath)
    initial_cols = X.shape[1]
    
    X.replace('None', np.nan, inplace=True)
    
    cols_to_drop = set()
    
    for col in X.columns:
        # 1. Drop Constant Columns
        if X[col].nunique(dropna=True) <= 1:
            cols_to_drop.add(col)
            continue
            
        # 2. Drop High-Null Columns
        null_pct = X[col].isna().mean()
        if null_pct > 0.90:
            cols_to_drop.add(col)

    # Perform the drop
    cols_to_drop = list(cols_to_drop)
    X_reduced = X.drop(columns=cols_to_drop)
    
    final_cols = X_reduced.shape[1]
    removed_count = initial_cols - final_cols
    
    print("\n--- Summary ---")
    print(f"Initial columns: {initial_cols}")
    print(f"Removed columns: {removed_count}")
    print(f"Remaining columns: {final_cols}")
    
    print("\n--- Removed Columns ---")
    print(cols_to_drop)
    
    # Save back to X_features.csv (overwriting it for the next ML step)
    X_reduced.to_csv(filepath, index=False)
    print(f"\nSuccessfully reduced {filepath}")

if __name__ == '__main__':
    filepath = 'data/processed/X_features.csv'
    reduce_features(filepath)
