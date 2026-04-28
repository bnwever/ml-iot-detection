import pandas as pd

def clean_feature_matrix(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    cols_to_drop = []
    
    for col in df.columns:
        # 4. Remove Unique Identifiers
        if col == 'uid':
            cols_to_drop.append(col)
            continue
            
        # 1. Identify and Remove Raw IP Columns (http_host)
        if col == 'http_host':
            cols_to_drop.append(col)
            continue
            
        # 1 & 2. Identify Raw IP columns but Preserve Aggregated Features
        if ('id.orig_h' in col or 'id.resp_h' in col):
            if not col.endswith('_nunique'):
                cols_to_drop.append(col)

    # Perform the drop
    df_cleaned = df.drop(columns=cols_to_drop)
    
    # 5. Handling the Target
    if '_device' in df_cleaned.columns:
        y = df_cleaned['_device']
        X = df_cleaned.drop(columns=['_device'])
    else:
        X = df_cleaned
        y = None
        
    print("\n--- Summary ---")
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
    print(f"Feature matrix (X) shape: {X.shape}")
    if y is not None:
        print(f"Target vector (y) shape: {y.shape}")
        
    print("\n--- Remaining Features in X ---")
    print(X.columns.tolist())
    
    return X, y

if __name__ == '__main__':
    # Trying data/processed/feature_matrix.csv based on our previous generation step
    filepath = 'data/processed/feature_matrix.csv'
    try:
        X, y = clean_feature_matrix(filepath)
        # We can save these to disk
        X.to_csv('data/processed/X_features.csv', index=False)
        y.to_csv('data/processed/y_target.csv', index=False)
        print("\nSuccessfully saved clean X_features.csv and y_target.csv")
    except FileNotFoundError:
        # Fallback if it is strictly inside logs/ folder
        filepath = 'data/processed/logs/feature_matrix.csv'
        X, y = clean_feature_matrix(filepath)
