import os
import glob
import pandas as pd
import hashlib
from scapy.all import rdpcap, IP, TCP, UDP, Ether
import argparse

# Configuration
RAW_DATA_DIR = 'data/raw/captures_IoT-Sentinel'
PROCESSED_DATA_DIR = 'data/processed'
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'iot_sentinel.parquet')
FLOW_TIMEOUT = 120  # seconds

def process_pcap(pcap_file, device_name, setup_id, device_mac):
    packets = rdpcap(pcap_file)
    flows = {}
    packet_data = []

    for pkt in packets:
        pkt_info = {
            'device_name': device_name,
            'setup_id': setup_id,
            'device_mac': device_mac,
            'time': float(pkt.time),
            'length': len(pkt),
            'src_mac': None,
            'dst_mac': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'protocol': None,
            'tcp_flags': None,
            'flow_id': None
        }
        
        if Ether in pkt:
            pkt_info['src_mac'] = pkt[Ether].src
            pkt_info['dst_mac'] = pkt[Ether].dst
            
        if IP in pkt:
            pkt_info['src_ip'] = pkt[IP].src
            pkt_info['dst_ip'] = pkt[IP].dst
            pkt_info['protocol'] = pkt[IP].proto
            
            if TCP in pkt:
                pkt_info['src_port'] = pkt[TCP].sport
                pkt_info['dst_port'] = pkt[TCP].dport
                pkt_info['tcp_flags'] = str(pkt[TCP].flags)
            elif UDP in pkt:
                pkt_info['src_port'] = pkt[UDP].sport
                pkt_info['dst_port'] = pkt[UDP].dport
                
        # Flow ID
        if pkt_info['src_ip'] and pkt_info['dst_ip']:
            p1 = f"{pkt_info['src_ip']}:{pkt_info['src_port']}"
            p2 = f"{pkt_info['dst_ip']}:{pkt_info['dst_port']}"
            endpoints = sorted([p1, p2])
            flow_key = f"{endpoints[0]}-{endpoints[1]}-{pkt_info['protocol']}"
            
            if flow_key in flows:
                last_time, flow_hash = flows[flow_key]
                if pkt_info['time'] - last_time > FLOW_TIMEOUT:
                    flow_hash = hashlib.md5(f"{flow_key}-{pkt_info['time']}".encode()).hexdigest()
            else:
                flow_hash = hashlib.md5(f"{flow_key}-{pkt_info['time']}".encode()).hexdigest()
                
            flows[flow_key] = (pkt_info['time'], flow_hash)
            pkt_info['flow_id'] = flow_hash
            
        packet_data.append(pkt_info)
        
    return packet_data

def main():
    all_packets = []
    
    # Ensure processed dir exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    print(f"Reading devices from {RAW_DATA_DIR}...")
    
    # Iterate over device directories
    for device_dir in os.listdir(RAW_DATA_DIR):
        device_path = os.path.join(RAW_DATA_DIR, device_dir)
        if not os.path.isdir(device_path):
            continue
            
        mac_file = os.path.join(device_path, '_iotdevice-mac.txt')
        device_mac = None
        if os.path.exists(mac_file):
            with open(mac_file, 'r') as f:
                device_mac = f.read().strip()
                
        pcap_files = glob.glob(os.path.join(device_path, '*.pcap'))
        print(f"Processing device: {device_dir} ({len(pcap_files)} PCAPs)")
        
        for pcap_file in pcap_files:
            # Extract setup_id from filename e.g. Setup-A-1-STA.pcap
            base_name = os.path.basename(pcap_file)
            parts = base_name.split('-')
            setup_id = "-".join(parts[1:-1]) if len(parts) > 2 else "Unknown"
            
            try:
                packets = process_pcap(pcap_file, device_dir, setup_id, device_mac)
                all_packets.extend(packets)
            except Exception as e:
                print(f"Error processing {pcap_file}: {e}")
                
    print(f"Total packets extracted: {len(all_packets)}")
    if len(all_packets) > 0:
        df = pd.DataFrame(all_packets)
        print(f"Saving to {OUTPUT_FILE}...")
        df.to_parquet(OUTPUT_FILE, index=False)
        print("Done!")
    else:
        print("No packets found.")

if __name__ == '__main__':
    main()
