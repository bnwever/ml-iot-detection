import json
import os
from collections import defaultdict

input_file = 'data/processed/combined_dataset.json'
output_dir = 'data/processed/'

def main():
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Group by _log_type
    grouped_data = defaultdict(list)
    for item in data:
        log_type = item.get('_log_type', 'unknown')
        grouped_data[log_type].append(item)

    # Save to separate files
    print(f"Found {len(grouped_data)} log types: {list(grouped_data.keys())}")
    for log_type, items in grouped_data.items():
        output_file = os.path.join(output_dir, f'{log_type}_dataset.json')
        print(f"Writing {len(items)} records to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(items, f, indent=2)
            
    print("Done!")

if __name__ == '__main__':
    main()
