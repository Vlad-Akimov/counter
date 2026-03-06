import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import argrelextrema

from ma1 import print_yellow_columns

def plot_distribution_for_columns(input_txt_path, output_dir="res/plots"):
    yellow_values = print_yellow_columns(input_txt_path, degud=False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    height, num_columns = yellow_values.shape
    # num_columns = 3
    print(f"\n{'='*60}")
    print(f"BUILDING DISTRIBUTION PLOTS FOR {num_columns} COLUMNS")
    print(f"{'='*60}")
    
    row_numbers = np.arange(1, height + 1)
    
    all_means = []
    all_mins = []
    all_maxs = []
    all_stds = []
    all_minima_counts = []
    all_maxima_counts = []
    
    for col_idx in range(num_columns):
        col_data = yellow_values[:, col_idx]
        
        mean_val = np.mean(col_data)
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        std_val = np.std(col_data)
        
        local_min_indices = argrelextrema(col_data, np.less)[0]
        local_max_indices = argrelextrema(col_data, np.greater)[0]
        num_minima = len(local_min_indices)
        num_maxima = len(local_max_indices)
        
        all_means.append(mean_val)
        all_mins.append(min_val)
        all_maxs.append(max_val)
        all_stds.append(std_val)
        all_minima_counts.append(num_minima)
        all_maxima_counts.append(num_maxima)
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(row_numbers, col_data, 'b-', linewidth=1.5, alpha=0.7, label='Values')
        
        if num_minima > 0:
            plt.scatter(row_numbers[local_min_indices], col_data[local_min_indices], 
                       color='red', s=50, zorder=5, label=f'Minima: {num_minima}')
        if num_maxima > 0:
            plt.scatter(row_numbers[local_max_indices], col_data[local_max_indices], 
                       color='green', s=50, zorder=5, label=f'Maxima: {num_maxima}')
        
        plt.title(f'Column {col_idx + 1} - Line Distribution', fontsize=14)
        plt.xlabel('Row Number', fontsize=12)
        plt.ylabel('Average Value', fontsize=12)
        plt.ylim(0, 255)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'column_{col_idx+1}_line.png')
        # plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nColumn {col_idx + 1}:")
        print(f"  Mean: {mean_val:.2f}")
        print(f"  Min: {min_val:.2f}")
        print(f"  Max: {max_val:.2f}")
        print(f"  Std: {std_val:.2f}")
        print(f"  Local minima: {num_minima}")
        print(f"  Local maxima: {num_maxima}")
        print(f"  Plot saved: {output_file}")
    
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, num_columns))
    
    for col_idx in range(num_columns):
        col_data = yellow_values[:, col_idx]
        plt.plot(row_numbers, col_data, color=colors[col_idx], 
                linewidth=1.5, alpha=0.7, label=f'Column {col_idx + 1}')
    
    plt.title('All Yellow Columns Comparison', fontsize=16)
    plt.xlabel('Row Number', fontsize=12)
    plt.ylabel('Average Value', fontsize=12)
    plt.ylim(0, 255)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    comparison_file = os.path.join(output_dir, 'all_columns_comparison.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS FOR ALL COLUMNS")
    print(f"{'='*60}")
    print(f"{'Column':<10} {'Mean':<10} {'Min':<10} {'Max':<10} {'Std':<10} {'Minima':<8} {'Maxima':<8}")
    print(f"{'-'*64}")
    
    for i in range(num_columns):
        print(f"{i+1:<10} {all_means[i]:<10.2f} {all_mins[i]:<10.2f} {all_maxs[i]:<10.2f} "
              f"{all_stds[i]:<10.2f} {all_minima_counts[i]:<8} {all_maxima_counts[i]:<8}")
    
    print(f"{'='*60}")
    print(f"OVERALL STATISTICS:")
    print(f"  Mean of all columns: {np.mean(all_means):.2f}")
    print(f"  Min of all columns: {np.min(all_mins):.2f}")
    print(f"  Max of all columns: {np.max(all_maxs):.2f}")
    print(f"  Average minima per column: {np.mean(all_minima_counts):.2f}")
    print(f"  Average maxima per column: {np.mean(all_maxima_counts):.2f}")
    print(f"  Total minima across all columns: {sum(all_minima_counts)}")
    print(f"  Total maxima across all columns: {sum(all_maxima_counts)}")
    print(f"{'='*60}")
    print(f"Comparison plot saved: {comparison_file}")
    
    return yellow_values

if __name__ == "__main__":
    block_size = 4
    n = 33
    
    # input_file = f"res/txt/output_q{n}_{block_size}{block_size}.txt"
    input_file = f"res/txt/sliding_q{n}_w8.txt"
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
    else:
        plot_distribution_for_columns(input_file)