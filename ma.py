import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import argrelextrema

def analyze_matrix(input_txt_path, output_plot_path):
    """
    Analysis of matrix from text file:
    - calculating row means ("yellow" columns in Excel)
    - plotting distribution of these means
    - counting local extrema
    """
    # Read matrix from file
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        matrix = []
        for line in f:
            row = [int(x) for x in line.strip().split()]
            matrix.append(row)
    
    matrix = np.array(matrix)
    print(f"Matrix shape: {matrix.shape}")
    
    # Calculate mean for each row
    row_means = np.mean(matrix, axis=1)
    
    # Find local minima and maxima
    local_min_indices = argrelextrema(row_means, np.less)[0]
    local_max_indices = argrelextrema(row_means, np.greater)[0]
    
    n_minima = len(local_min_indices)
    n_maxima = len(local_max_indices)
    
    print(f"Local minima: {n_minima}")
    print(f"Local maxima: {n_maxima}")
    print(f"Total extrema: {n_minima + n_maxima}")
    
    # Create directory for saving
    output_dir = os.path.dirname(output_plot_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build distribution plot
    plt.figure(figsize=(15, 7))
    
    # Row numbers for X axis
    row_numbers = np.arange(1, len(row_means) + 1)
    
    # Distribution plot
    plt.plot(row_numbers, row_means, 'b-', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Row Number')
    plt.ylabel('Average Brightness Value (0-255)')
    plt.title('Distribution of Row Mean Values')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved: {output_plot_path}")
    
    return n_minima, n_maxima

if __name__ == "__main__":
    # Parameters
    block_size = 10
    n = 59
    
    input_file = f"res/txt/output_q{n}_{block_size}{block_size}.txt"
    output_plot = f"res/plots/distribution_q{n}_rows.png"
    
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        print("Please run cb.py first to create the matrix file")
    else:
        # Run analysis
        n_min, n_max = analyze_matrix(input_file, output_plot)
