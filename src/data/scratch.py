import pandas as pd
import glob
import json
import os

input_dir = "data/processed/logs"
all_files = glob.glob(os.path.join(input_dir, "*.json"))

dataframes = {}
for file in all_files:
    log_type = os.path.basename(file).replace("_dataset.json", "")
    df = pd.read_json(file)
    print(f"{log_type}: {df.columns.tolist()[:5]}")
