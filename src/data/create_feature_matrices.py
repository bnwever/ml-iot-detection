import pandas as pd
import glob
import os
import warnings

def create_feature_matrices():
    warnings.filterwarnings('ignore')
    
    input_dir = "data/processed/logs"
    output_dir = "data/processed/feature_matrices"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not all_files:
        print(f"No JSON files found in {input_dir}")
        return
        
    total_logs = 0
    
    for file in all_files:
        log_type = os.path.basename(file).replace("_dataset.json", "")
        df = pd.read_json(file)
        
        # Track total logs to ensure we keep all 2797
        total_logs += len(df)
        
        # Columns to drop for IPs
        cols_to_drop = [
            col for col in df.columns 
            if 'id.orig_h' in col or 'id.resp_h' in col or col == 'http_host' or col == 'host'
        ]
        
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Clean up metadata columns
        if '_device' in df.columns:
            df = df.rename(columns={'_device': 'device'})
            
        if '_log_type' in df.columns:
            df = df.drop(columns=['_log_type'])
            
        if 'ts' in df.columns:
            df = df.drop(columns=['ts'])
            
        # Set uid as index if it exists
        if 'uid' in df.columns:
            df = df.set_index('uid')
            
        output_path = os.path.join(output_dir, f"{log_type}_feature_matrix.csv")
        df.to_csv(output_path)
        print(f"Saved {log_type} feature matrix: {df.shape} to {output_path}")

    print(f"\nTotal logs preserved across all feature matrices: {total_logs}")

if __name__ == "__main__":
    create_feature_matrices()
