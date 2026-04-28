import pandas as pd
import glob
import os
import warnings

def create_feature_matrix():
    warnings.filterwarnings('ignore')
    
    input_dir = "data/processed/logs"
    output_path = "data/processed/feature_matrix.csv"
    
    all_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not all_files:
        print(f"No JSON files found in {input_dir}")
        return
        
    dataframes = []
    
    for file in all_files:
        log_type = os.path.basename(file).replace("_dataset.json", "")
        df = pd.read_json(file)
        
        if 'uid' not in df.columns:
            continue
            
        # Set uid as index, take first if duplicates (to ensure 1-to-1 merge)
        df = df.groupby('uid').first()
        
        # Prefix columns with log_type to prevent collisions
        df = df.add_prefix(f"{log_type}_")
        dataframes.append(df)
    
    print("Combining datasets...")
    # Combine all dataframes on uid
    combined_df = pd.concat(dataframes, axis=1)
    
    # Columns to drop
    cols_to_drop = [
        col for col in combined_df.columns 
        if 'id.orig_h' in col or 'id.resp_h' in col or col == 'http_host'
    ]
    
    combined_df = combined_df.drop(columns=cols_to_drop)
    
    # Consolidate device columns into one
    device_cols = [col for col in combined_df.columns if col.endswith('_device')]
    if device_cols:
        combined_df['device'] = combined_df[device_cols].bfill(axis=1).iloc[:, 0]
        combined_df = combined_df.drop(columns=device_cols)
        
    # Consolidate log_type columns or drop them since they are not useful features
    log_type_cols = [col for col in combined_df.columns if col.endswith('_log_type')]
    combined_df = combined_df.drop(columns=log_type_cols)
    
    # Drop timestamp columns as they are usually not generalizable features
    ts_cols = [col for col in combined_df.columns if col.endswith('_ts')]
    combined_df = combined_df.drop(columns=ts_cols)
    
    print(f"Combined shape: {combined_df.shape}")
    print(f"Columns dropped for IPs: {cols_to_drop}")
    
    combined_df.to_csv(output_path)
    print(f"Feature matrix saved to {output_path}")

if __name__ == "__main__":
    create_feature_matrix()
