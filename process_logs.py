import os
import json
import glob

base_dir = "data/interim/raw/captures_IoT-Sentinel"
output_dir = "data/processes"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_records = []

for root, dirs, files in os.walk(base_dir):
    device_name = os.path.basename(root)
    if device_name == "captures_IoT-Sentinel" or not files:
        continue
    
    for file in files:
        if not file.endswith(".log"):
            continue
            
        log_type = file.replace(".log", "")
        file_path = os.path.join(root, file)
        
        fields = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#fields"):
                    # Tab separated
                    fields = line.split("\t")[1:]
                elif line.startswith("#"):
                    continue
                else:
                    if not fields:
                        continue
                    values = line.split("\t")
                    record = dict(zip(fields, values))
                    record["_device"] = device_name
                    record["_log_type"] = log_type
                    all_records.append(record)

output_file = os.path.join(output_dir, "combined_dataset.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_records, f, indent=2)

print(f"Processed {len(all_records)} records into {output_file}")
