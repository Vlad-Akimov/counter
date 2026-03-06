import numpy as np
import os

def print_yellow_columns(input_txt_path, degud=True):
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    matrix = []
    for line in lines:
        row = [int(x) for x in line.strip().split()]
        matrix.append(row)
    
    matrix = np.array(matrix, dtype=np.uint8)
    height, width = matrix.shape
    print(f"Original matrix: {height} x {width}")
    
    new_width = width - (width % 8)
    matrix_trimmed = matrix[:, :new_width]
    
    yellow_values = matrix_trimmed.reshape(height, new_width // 8, 8).mean(axis=2)
    
    if degud:
        # Print all columns with 2 decimal places
        for col_idx in range(yellow_values.shape[1]):
            print(f"\nColumn {col_idx + 1}:")
            # Take column and round to 2 decimals
            column_data = yellow_values[:, col_idx]
            rounded_data = [round(val) for val in column_data]
            
            # Print values
            print("[", end="")
            for i, val in enumerate(rounded_data):
                if i > 0:
                    print(", ", end="")
                print(f"{val}", end="")
            print("]")
    
    return yellow_values

if __name__ == "__main__":
    block_size = 10
    n = 59
    
    input_file = f"res/txt/output_q{n}_{block_size}{block_size}.txt"
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
    else:
        yellow_columns = print_yellow_columns(input_file)