from PIL import Image
import numpy as np
from compression_pipeline import compress_image, decompress_image
from quantization_matrices import quantization_matrices
from dct_quantization import compute_dct_blocks

# Load grayscale image
image = Image.open('Data Set-1/101_ObjectCategories/airplanes/image_0001.jpg').convert('L')

# Ensure the dimensions are multiples of 8
new_width = (image.width // 8 + 1) * 8  # nearest multiple of 8
new_height = (image.height // 8 + 1) * 8  # nearest multiple of 8
print(f"Resized to: {new_width} and {new_height}")

# Pad the image to the next multiple of 8 (using Image object)

# padded_img = image.resize((new_width, new_height))
padded_matrix = np.zeros((new_height, new_width), dtype=np.int32)

# Copy the original image matrix into the padded matrix
padded_matrix[:image.height, :image.width] = image

# Convert the padded image to matrix form (as 2D list of pixel values)
image_matrix = np.array(padded_matrix, dtype=np.int32)
# Loop through all quantization matrices
for name, quant_matrix in quantization_matrices.items():
    print(f"Processing with {name} matrix...")

    # Split image matrix into 8x8 blocks
    # blocks = compute_dct_blocks(image_matrix)
    # print(blocks[7])

    # Compress the image
    encoded_blocks, huffman, shape = compress_image(image_matrix, quant_matrix)

    # Decompress the image
    restored_image = decompress_image(encoded_blocks, huffman, quant_matrix, shape)

    # Convert the restored image back to PIL image for saving
    restored_image_pil = Image.fromarray(restored_image)

    # Save the restored image with a unique name
    restored_image_pil.save(f"restored_{name}.jpg")

    print(f"Saved restored_{name}.jpg")

print("All matrices processed and images saved.")