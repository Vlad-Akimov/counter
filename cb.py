from PIL import Image
import numpy as np


def convert_image(input_path, output_txt_path, block_size):
    img = Image.open(input_path)

    img_gray = img.convert("L")
    img_array = np.array(img_gray, dtype=np.uint8)

    height, width = img_array.shape

    new_height = height - (height % block_size)
    new_width = width - (width % block_size)

    img_array = img_array[:new_height, :new_width]

    grouped = img_array.reshape(new_height // block_size, block_size,
                                new_width // block_size, block_size).mean(axis=(1, 3))

    grouped = grouped.astype(np.uint8)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for row in grouped:
            line = " ".join(f"{val:3d}" for val in row)
            f.write(line + "\n")

    print(f"Matrix: {grouped.shape}")
    print(f"file: {output_txt_path}")


if __name__ == "__main__":
    block_size = 3
    n = 60
    input_image = f"res/photos/q{n}.jpg"
    output_file = f"res/txt/output_q{n}_{block_size}{block_size}.txt"

    convert_image(input_image, output_file, block_size)