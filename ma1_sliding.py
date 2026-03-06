import numpy as np
import os

def process_with_sliding_window(input_txt_path, window_size=8, degud=True):
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    matrix = []
    for line in lines:
        row = [int(x) for x in line.strip().split()]
        matrix.append(row)
    
    matrix = np.array(matrix, dtype=np.uint8)
    height, width = matrix.shape
    print(f"Original matrix: {height} x {width}")
    
    new_width = width - (width % window_size)
    matrix_trimmed = matrix[:, :new_width]
    non_overlap_result = matrix_trimmed.reshape(height, new_width // window_size, window_size).mean(axis=2)
    
    print(f"\n{'='*60}")
    print("PART 1: NON-OVERLAPPING AVERAGING")
    print(f"{'='*60}")
    print(f"Result shape: {non_overlap_result.shape}")
    
    num_windows = width - window_size + 1
    sliding_result = np.zeros((height, num_windows))
    
    for start_col in range(num_windows):
        end_col = start_col + window_size
        window_data = matrix[:, start_col:end_col]
        sliding_result[:, start_col] = window_data.mean(axis=1)
    
    print(f"\n{'='*60}")
    print("PART 2: SLIDING WINDOW AVERAGING (shift by 1)")
    print(f"{'='*60}")
    print(f"Result shape: {sliding_result.shape}")
    print(f"Number of windows: {num_windows}")
    
    if degud:
        print("\n" + "="*60)
        print("NON-OVERLAPPING AVERAGING RESULTS")
        print("="*60)
        for col_idx in range(non_overlap_result.shape[1]):
            print(f"\nColumn {col_idx + 1} (non-overlapping):")
            column_data = non_overlap_result[:, col_idx]
            rounded_data = [round(val) for val in column_data]
            
            print("[", end="")
            for i, val in enumerate(rounded_data):
                if i > 0:
                    print(", ", end="")
                print(f"{val}", end="")
            print("]")
        
        print("\n" + "="*60)
        print("SLIDING WINDOW AVERAGING RESULTS (shift by 1)")
        print("="*60)
        
        print("\nFirst 5 windows:")
        for col_idx in range(min(5, sliding_result.shape[1])):
            print(f"\nWindow {col_idx + 1} (columns {col_idx+1}-{col_idx+window_size}):")
            column_data = sliding_result[:, col_idx]
            rounded_data = [round(val) for val in column_data]
            
            print("[", end="")
            for i, val in enumerate(rounded_data):
                if i > 0:
                    print(", ", end="")
                print(f"{val}", end="")
            print("]")
        
        if sliding_result.shape[1] > 5:
            print("\n...")
            print(f"\nLast 5 windows:")
            for col_idx in range(sliding_result.shape[1]-5, sliding_result.shape[1]):
                print(f"\nWindow {col_idx + 1} (columns {col_idx+1}-{col_idx+window_size}):")
                column_data = sliding_result[:, col_idx]
                rounded_data = [round(val) for val in column_data]
                
                print("[", end="")
                for i, val in enumerate(rounded_data):
                    if i > 0:
                        print(", ", end="")
                    print(f"{val}", end="")
                print("]")
    
    print("\n" + "="*60)
    print("STATISTICS COMPARISON")
    print("="*60)
    print(f"{'Method':<25} {'Shape':<15} {'Mean':<10} {'Std':<10}")
    print(f"{'-'*60}")
    
    non_overlap_mean = np.mean(non_overlap_result)
    non_overlap_std = np.std(non_overlap_result)
    print(f"{'Non-overlapping':<25} {str(non_overlap_result.shape):<15} {non_overlap_mean:<10.2f} {non_overlap_std:<10.2f}")
    
    sliding_mean = np.mean(sliding_result)
    sliding_std = np.std(sliding_result)
    print(f"{'Sliding window':<25} {str(sliding_result.shape):<15} {sliding_mean:<10.2f} {sliding_std:<10.2f}")
    
    mean_diff = abs(sliding_mean - non_overlap_mean)
    std_diff = abs(sliding_std - non_overlap_std)
    print(f"{'Difference':<25} {'':<15} {mean_diff:<10.2f} {std_diff:<10.2f}")
    
    return non_overlap_result, sliding_result

if __name__ == "__main__":
    block_size = 4
    n = 33
    
    input_file = f"res/txt/output_q{n}_{block_size}{block_size}.txt"
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        print("Please run cb.py first")
    else:
        non_overlap, sliding = process_with_sliding_window(input_file, window_size=8)
        
        output_dir = "res/txt"
        os.makedirs(output_dir, exist_ok=True)
        
        # non_overlap_file = os.path.join(output_dir, f"non_overlap_q{n}_w8.txt")
        # np.savetxt(non_overlap_file, non_overlap, fmt='%d', delimiter=' ')
        # print(f"\nNon-overlapping result saved to: {non_overlap_file}")
        
        sliding_file = os.path.join(output_dir, f"sliding_q{n}_w8.txt")
        np.savetxt(sliding_file, sliding, fmt='%d', delimiter=' ')
        print(f"Sliding window result saved to: {sliding_file}")