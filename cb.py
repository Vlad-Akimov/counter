from PIL import Image
import numpy as np


def convert_image(input_path, output_txt_path, block_size):
    img = Image.open(input_path)

    img_gray = img.convert("L")
    img_array = np.array(img_gray, dtype=np.uint8)

    height, width = img_array.shape

    vertical_block = 8
    
    new_height = height - (height % vertical_block)
    new_width = width - (width % block_size)

    img_array = img_array[:new_height, :new_width]

    grouped = img_array.reshape(new_height // vertical_block, vertical_block,
                               new_width // block_size, block_size)
    
    grouped = grouped.mean(axis=(1, 3))

    grouped = grouped.astype(np.uint8)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for row in grouped:
            line = " ".join(f"{val:3d}" for val in row)
            f.write(line + "\n")

    print(f"\nResult matrix: {grouped.shape[0]} rows x {grouped.shape[1]} columns")
    print(f"  - Each row represents {vertical_block} original rows")
    print(f"  - Each column represents {block_size} original columns")
    print(f"File saved: {output_txt_path}")


if __name__ == "__main__":
    block_size = 4
    n = 33
    input_image = f"res/photos/q{n}.jpg"
    output_file = f"res/txt/output_q{n}_{block_size}{block_size}.txt"

    convert_image(input_image, output_file, block_size)