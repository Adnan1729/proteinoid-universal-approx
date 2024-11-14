# analyze_complexity.py (scripts/analyze_complexity.py) - continued

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_spike_data
from src.analysis.complexity import (F1, F2, find_point_less_than,
                                   find_points_with_same_digit, calculate_complexity_metrics)
from src.analysis.visualization import plot_spike_trains, plot_complexity_metrics
from src.utils.metrics import calculate_meta_metric
import argparse
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    # Load data
    dataframes = load_spike_data(args.data_path)
    
    # Plot spike trains
    fig_spike_trains = plot_spike_trains(dataframes)
    fig_spike_trains.savefig(os.path.join(args.output_dir, 'spike_trains.png'))
    
    # Calculate complexity metrics for each dataset
    all_metrics = []
    for i, df in enumerate(dataframes, 1):
        G = nx.Graph()
        t_values = np.arange(1, 21)
        x_t_values = F1(t_values)
        selected_points = [find_point_less_than(df, x) for x in x_t_values]
        
        for point in selected_points:
            if pd.notna(point):
                G.add_node(point)
                d = F2(point)
                connected_points = find_points_with_same_digit(df, point, d)
                for connected_point in connected_points:
                    if connected_point != point:
                        G.add_node(connected_point)
                        G.add_edge(point, connected_point)
        
        # Calculate metrics for this dataset
        metrics = calculate_complexity_metrics(G)
        metrics['Dataset'] = f'Dataset {i}'
        all_metrics.append(metrics)
        
        # Print basic statistics for this dataset
        print(f"\nDataset {i} Statistics:")
        print(f"Number of spikes: {df['Spike'].sum()}")
        print(f"Total duration: {df['Time'].max() - df['Time'].min():.2f} seconds")
        print(f"Average inter-spike interval: {df['Time'].diff().mean():.4f} seconds")
        
        # Print transformation analysis
        print(f"\nDataset {i} Transformation Analysis:")
        print("F1 values (x(t)):")
        print(", ".join(f"{x:.2f}" for x in x_t_values))
        
        print("\nSelected points from dataset:")
        print(", ".join(f"{x:.6f}" if pd.notna(x) else "N/A" for x in selected_points))
        
        d_values = [F2(x) if pd.notna(x) else "N/A" for x in selected_points]
        print("\nFirst significant digits after decimal point (d):")
        print(", ".join(str(int(d)) if pd.notna(d) and d != "N/A" else "N/A" for d in d_values))
        
        print("\nNumber of connections for each selected point:")
        for point, d in zip(selected_points, d_values):
            if pd.notna(point) and d != "N/A":
                connected_points = find_points_with_same_digit(df, point, d)
                print(f"Point {point:.6f} (d={int(d)}): {len(connected_points)} connections")
    
    # Create DataFrame with all metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate meta-metric
    meta_metric = calculate_meta_metric(metrics_df.drop('Dataset', axis=1))
    metrics_df['meta_metric'] = meta_metric
    
    # Sort DataFrame by meta-metric
    metrics_df = metrics_df.sort_values('meta_metric', ascending=False)
    
    # Format the DataFrame for display
    display_df = metrics_df.copy()
    for col in display_df.columns:
        if col != 'Dataset':
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
    
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df = display_df.set_index('Rank')
    
    # Reorder columns to have Dataset name first
    cols = ['Dataset'] + [col for col in display_df.columns if col != 'Dataset']
    display_df = display_df[cols]
    
    # Save metrics to CSV
    metrics_output_path = os.path.join(args.output_dir, 'complexity_metrics.csv')
    display_df.to_csv(metrics_output_path)
    print(f"\nComplexity metrics saved to: {metrics_output_path}")
    
    # Plot complexity metrics
    fig_metrics = plot_complexity_metrics(metrics_df)
    fig_metrics.savefig(os.path.join(args.output_dir, 'complexity_metrics.png'))
    print(f"Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze spike train complexity')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to spike data CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save output files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
