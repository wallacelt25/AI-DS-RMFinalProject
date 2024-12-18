import os
import pyshark
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import cProfile
import time

def compute_flow_features(capture):
    features = []
    flows = {}

    for packet_num, packet in enumerate(capture, start=1):
        try:
            # Only process TCP/UDP packets
            if not hasattr(packet, 'transport_layer') or packet.transport_layer not in ['TCP', 'UDP']:
                continue

            if not hasattr(packet, 'length') or not hasattr(packet, 'ip'):
                continue

            # Extract flow identifiers
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            src_port = packet[packet.transport_layer].srcport
            dst_port = packet[packet.transport_layer].dstport

            forward_key = (src_ip, dst_ip, src_port, dst_port)
            backward_key = (dst_ip, src_ip, dst_port, src_port)

            # Initialize flow if not already present
            if forward_key not in flows and backward_key not in flows:
                flows[forward_key] = {
                    'fwd': 0, 'bwd': 0,
                    'fwd_lengths': [], 'bwd_lengths': [],
                    'fwd_timestamps': [], 'bwd_timestamps': [],
                    'fwd_window_size': None, 'bwd_window_size': None,
                    'timestamps': [],
                    'start_time': float(packet.sniff_timestamp)
                }

            # Determine direction
            packet_length = int(packet.length)
            flow = flows[forward_key] if forward_key in flows else flows[backward_key]

            if forward_key in flows:
                flow['fwd'] += 1
                flow['fwd_lengths'].append(packet_length)
                flow['fwd_timestamps'].append(float(packet.sniff_timestamp))
                if packet.transport_layer == 'TCP' and hasattr(packet.tcp, 'window_size'):
                    if flow['fwd_window_size'] is None:
                        flow['fwd_window_size'] = int(packet.tcp.window_size)
            else:
                flow['bwd'] += 1
                flow['bwd_lengths'].append(packet_length)
                flow['bwd_timestamps'].append(float(packet.sniff_timestamp))
                if packet.transport_layer == 'TCP' and hasattr(packet.tcp, 'window_size'):
                    if flow['bwd_window_size'] is None:
                        flow['bwd_window_size'] = int(packet.tcp.window_size)

            # Record timestamps for IAT calculations
            flow['timestamps'].append(float(packet.sniff_timestamp))
            flow['end_time'] = float(packet.sniff_timestamp)

        except Exception as e:
            print(f"Error processing packet #{packet_num}: {e}")
            continue

    # Compute features for each flow
    for flow_key, flow_data in flows.items():
        all_lengths = flow_data['fwd_lengths'] + flow_data['bwd_lengths']
        duration = flow_data['end_time'] - flow_data['start_time'] if 'end_time' in flow_data else 0

        # Compute IAT features
        iats = np.diff(flow_data['timestamps']) if len(flow_data['timestamps']) > 1 else np.array([])
        flow_iat_min = np.min(iats) if len(iats) > 0 else 0
        flow_iat_max = np.max(iats) if len(iats) > 0 else 0

        # Compute Variance
        packet_length_variance = np.var(all_lengths) if all_lengths else 0

        # Compute additional features
        features.append({
            'Init_Win_bytes_forward': flow_data['fwd_window_size'] if flow_data['fwd_window_size'] else 0,
            'Init_Win_bytes_backward': flow_data['bwd_window_size'] if flow_data['bwd_window_size'] else 0,
            'Flow IAT Min': flow_iat_min,
            'Flow Duration': duration,
            'Total Length of Fwd Packets': sum(flow_data['fwd_lengths']),
            'Packet Length Variance': packet_length_variance,
            'Bwd Packet Length Max': max(flow_data['bwd_lengths']) if flow_data['bwd_lengths'] else 0,
            'Avg Fwd Segment Size': np.mean(flow_data['fwd_lengths']) if flow_data['fwd_lengths'] else 0,
            'Max Packet Length': max(all_lengths) if all_lengths else 0,
            'Flow IAT Max': flow_iat_max,
            'Fwd Packet Length Mean': np.mean(flow_data['fwd_lengths']) if flow_data['fwd_lengths'] else 0,
            'Subflow Fwd Bytes': sum(flow_data['fwd_lengths']),
            'Bwd Packet Length Mean': np.mean(flow_data['bwd_lengths']) if flow_data['bwd_lengths'] else 0,
            'Avg Bwd Segment Size': np.mean(flow_data['bwd_lengths']) if flow_data['bwd_lengths'] else 0,
            'Flow Packets/s': (flow_data['fwd'] + flow_data['bwd']) / duration if duration > 0 else 0,
            'Fwd Packet Length Max': max(flow_data['fwd_lengths']) if flow_data['fwd_lengths'] else 0,
            'Flow Bytes/s': sum(all_lengths) / duration if duration > 0 else 0,
            'Fwd Packet Length Std': np.std(flow_data['fwd_lengths']) if flow_data['fwd_lengths'] else 0,
            'Bwd Packet Length Std': np.std(flow_data['bwd_lengths']) if flow_data['bwd_lengths'] else 0,
            'Bwd Packet Length Min': min(flow_data['bwd_lengths']) if flow_data['bwd_lengths'] else 0
        })

    return pd.DataFrame(features)

def capture_and_save(interface, duration, output_file, csv_output_file):
    try:
        start_time = time.time()
        if not os.path.exists(output_file):
            print(f"{output_file} does not exist. Starting live capture...")
            capture = pyshark.LiveCapture(interface=interface, output_file=output_file, display_filter="tcp or udp")
            capture.sniff(timeout=duration)
            print(f"Capture complete. Packets saved to {output_file}.")
        else:
            print(f"{output_file} already exists. Skipping capture.")

        offline_capture = pyshark.FileCapture(output_file, display_filter="tcp or udp")
        print("Computing flow features...")
        features_df = compute_flow_features(offline_capture)
        if not features_df.empty:
            features_df.to_csv(csv_output_file, index=False)
            print(f"Flow features saved to {csv_output_file}.")
        else:
            print("No features were extracted; CSV is empty.")

        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    interface = "\\Device\\NPF_{33FA2EB5-A09A-4A46-B331-E960253C773A}"  # Replace with your active interface
    duration = 20
    output_file = "output.pcap"
    csv_output_file = "flow_features.csv"

    with cProfile.Profile() as profiler:
        capture_and_save(interface, duration, output_file, csv_output_file)
    profiler.print_stats()
