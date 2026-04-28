import pandas as pd
import glob
import os
import warnings

warnings.filterwarnings('ignore')

input_dir = "data/processed/logs"
all_files = glob.glob(os.path.join(input_dir, "*.json"))

dataframes = []

for file in all_files:
    log_type = os.path.basename(file).replace("_dataset.json", "")
    df = pd.read_json(file)
    
    if 'uid' not in df.columns:
        continue
        
    # Set uid as index, take first if duplicates
    df = df.groupby('uid').first()
    
    # Prefix columns with log_type
    df = df.add_prefix(f"{log_type}_")
    dataframes.append(df)

# Combine all dataframes
combined_df = pd.concat(dataframes, axis=1)

# Columns to drop
cols_to_drop = [col for col in combined_df.columns if 'id.orig_h' in col or 'id.resp_h' in col or col == 'http_host']
combined_df = combined_df.drop(columns=cols_to_drop)

print(f"Combined shape: {combined_df.shape}")
print(f"Columns dropped: {cols_to_drop}")

# Also if there are multiple device columns like conn__device, http__device, maybe consolidate them?
device_cols = [col for col in combined_df.columns if col.endswith('_device')]
if device_cols:
    combined_df['device'] = combined_df[device_cols].bfill(axis=1).iloc[:, 0]
    combined_df = combined_df.drop(columns=device_cols)

print(f"Final shape: {combined_df.shape}")
print(combined_df.head(2))

