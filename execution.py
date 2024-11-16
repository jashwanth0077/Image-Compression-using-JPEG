from PIL import Image
import numpy as np
from compression_pipeline import compress_image, decompress_image
from quantization_matrices import quantization_matrices

# Load grayscale image
image = np.array(Image.open('sample.jpg').convert('L'))

# Loop through all quantization matrices
for name, quant_matrix in quantization_matrices.items():
    print(f"Processing with {name} matrix...")

    # Compress the image
    encoded_blocks, huffman, shape = compress_image(image, quant_matrix)

    # Decompress the image
    restored_image = decompress_image(encoded_blocks, huffman, quant_matrix, shape)

    # Save the restored image with a unique name
    restored_image_pil = Image.fromarray(restored_image)
    restored_image_pil.save(f"restored_{name}.jpg")

    print(f"Saved restored_{name}.jpg")

print("All matrices processed and images saved.")
