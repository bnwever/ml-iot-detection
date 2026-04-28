import pandas as pd
import json
import glob
import os
import numpy as np

def load_log_data(log_dir):
    """Loads all JSON files into a dictionary of DataFrames."""
    dataframes = {}
    for file_path in glob.glob(os.path.join(log_dir, '*_dataset.json')):
        log_type = os.path.basename(file_path).replace('_dataset.json', '')
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        if 'uid' in df.columns:
            # Replace Zeek's default '-' for missing data with NaN
            df = df.replace('-', np.nan)
            dataframes[log_type] = df
            
    return dataframes

def build_feature_matrix(dataframes):
    """Transforms DataFrames into a 'One-Row-Per-UID' feature matrix."""
    
    # 1. Base Dataframe: Conflict Resolution - Retain correct _device label
    uids_devices = []
    for log_type, df in dataframes.items():
        if 'uid' in df.columns and '_device' in df.columns:
            uids_devices.append(df[['uid', '_device']])
            
    if not uids_devices:
        raise ValueError("No valid data with 'uid' and '_device' found.")
        
    base_df = pd.concat(uids_devices).drop_duplicates(subset=['uid']).set_index('uid')
    
    processed_dfs = []
    
    for log_type, df in dataframes.items():
        # Drop columns we don't need to join as features
        cols_to_keep = [c for c in df.columns if c not in ['uid', '_device', '_log_type']]
        
        # 2. High-Volume Log Compression
        if log_type in ['files', 'weird']:
            # Define numeric fields that need sum/mean
            numeric_cols = []
            if log_type == 'files':
                numeric_cols = ['seen_bytes', 'total_bytes', 'missing_bytes', 'overflow_bytes']
            
            # Convert numeric columns safely
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Define aggregations dynamically
            agg_funcs = {}
            for col in cols_to_keep:
                if col in numeric_cols:
                    agg_funcs[col] = ['sum', 'mean']
                else:
                    # Categorical fields get 'nunique'
                    agg_funcs[col] = ['nunique']
            
            # Count of sub-logs
            df['sub_log_count'] = 1
            agg_funcs['sub_log_count'] = ['sum']
            
            # Group by uid
            grouped = df.groupby('uid').agg(agg_funcs)
            
            # Flatten multi-level columns (e.g., ('seen_bytes', 'sum') -> 'files_seen_bytes_sum')
            grouped.columns = [f"{log_type}_{col[0]}_{col[1]}" for col in grouped.columns]
            processed_dfs.append(grouped)
            
        else:
            # 3. Horizontal Join for 1-to-1 logs
            df_subset = df[['uid'] + cols_to_keep].copy()
            # If there are accidental duplicates, keep the first one
            df_subset = df_subset.drop_duplicates(subset=['uid'])
            df_subset = df_subset.set_index('uid')
            # Prefix columns to avoid name collisions across log types
            df_subset.columns = [f"{log_type}_{col}" for col in df_subset.columns]
            processed_dfs.append(df_subset)

    # Combine all dataframes
    feature_matrix = base_df
    for p_df in processed_dfs:
        feature_matrix = feature_matrix.join(p_df, how='left')
        
    # 4. Handling Missing Data
    # Identify numeric and object columns to fill with appropriate placeholders
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    object_cols = feature_matrix.select_dtypes(include=['object']).columns

    # Fill numerical with 0 (or NaN if preferred for ML), categorical with 'None'
    feature_matrix[numeric_cols] = feature_matrix[numeric_cols].fillna(0)
    feature_matrix[object_cols] = feature_matrix[object_cols].fillna('None')
    
    return feature_matrix.reset_index()

if __name__ == '__main__':
    log_dir = 'data/processed/logs'
    print(f"Loading data from {log_dir}...")
    dfs = load_log_data(log_dir)
    
    print("Building feature matrix...")
    feature_matrix = build_feature_matrix(dfs)
    
    output_path = 'data/processed/feature_matrix.csv'
    feature_matrix.to_csv(output_path, index=False)
    print(f"Successfully generated matrix with shape {feature_matrix.shape}")
    print(f"Saved to {output_path}")
