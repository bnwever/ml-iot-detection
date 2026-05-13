import pandas as pd
import os

def extract_matrices():
    # File paths
    raw_data_path = 'data/raw/device_packets.parquet'
    output_x_path = 'data/processed/X_matrix.parquet'
    output_y_path = 'data/processed/y_matrix.parquet'
    
    print(f"Loading raw data from {raw_data_path}...")
    try:
        df = pd.read_parquet(raw_data_path)
    except Exception as e:
        print(f"Failed to load parquet, falling back to CSV: {e}")
        raw_data_path = 'data/raw/device_packets.csv'
        df = pd.read_csv(raw_data_path)

    # Defined features
    X_FEATURES = [
        'L4_tcp', 'L4_udp', 'L7_http', 'L7_https', 'port_class_src', 
        'port_class_dst', 'pck_size', 'ethernet_frame_size', 'ttl', 
        'total_length', 'protocol', 'source_port', 'dest_port', 'DNS_count', 
        'NTP_count', 'ARP_count', 'cnt', 'L3_ip_dst_count', 'most_freq_prot', 
        'most_freq_sport', 'most_freq_dport', 'sum_et', 'min_et', 'max_et', 
        'med_et', 'average_et', 'skew_et', 'kurt_et', 'var', 'q3', 'q1', 'iqr', 
        'sum_e', 'min_e', 'max_e', 'med', 'average', 'skew_e', 'kurt_e', 
        'var_e', 'q3_e', 'q1_e', 'iqr_e'
    ]

    Y_FEATURES = [
        'global_category', 'device', 'interaction_type', 'command'
    ]

    print("Extracting feature matrices...")
    
    # Check if all columns exist
    missing_x = [col for col in X_FEATURES if col not in df.columns]
    missing_y = [col for col in Y_FEATURES if col not in df.columns]
    
    if missing_x:
        print(f"Warning: Missing X features in dataset: {missing_x}")
        # Keep only available features
        X_FEATURES = [col for col in X_FEATURES if col in df.columns]
        
    if missing_y:
        print(f"Warning: Missing Y features in dataset: {missing_y}")
        Y_FEATURES = [col for col in Y_FEATURES if col in df.columns]

    X = df[X_FEATURES]
    y = df[Y_FEATURES]

    # Ensure output directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    print(f"Saving X matrix ({X.shape}) to {output_x_path}...")
    X.to_parquet(output_x_path, index=False)
    
    print(f"Saving y matrix ({y.shape}) to {output_y_path}...")
    y.to_parquet(output_y_path, index=False)
    
    print("Feature matrices successfully extracted and saved.")

if __name__ == '__main__':
    extract_matrices()
