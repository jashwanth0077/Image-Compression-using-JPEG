import numpy as np
from dct_quantization import *
from huffman_rle import *

def compress_image(image, quantization_matrix):
    dct_blocks = compute_dct_blocks(image)
    all_rle = []

    for block in dct_blocks:
        quantized_block = quantize_block(block, quantization_matrix)
        rle = run_length_encode(quantized_block)
        all_rle.extend(rle)

    values = [value for _, _, value in all_rle]
    huffman = HuffmanCoding()
    codes = huffman.encode(values)

    encoded_blocks = []
    for block in dct_blocks:
        quantized_block = quantize_block(block, quantization_matrix)
        rle = run_length_encode(quantized_block)
        encoded_blocks.append(huffman_encode_rle_with_global_codes(rle, huffman))

    return encoded_blocks, huffman, image.shape

def decompress_image(encoded_blocks, huffman, quantization_matrix, image_shape):
    block_size = 8
    h, w = image_shape
    restored_image = np.zeros((h, w))

    for i, encoded_block in enumerate(encoded_blocks):
        row, col = divmod(i, w // block_size)
        rle = huffman.decode(encoded_block)
        quantized_block = zigzag_to_block(rle)
        dequantized_block = dequantize_block(quantized_block, quantization_matrix)
        restored_image[row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size] = idct2(dequantized_block)

    return np.clip(restored_image, 0, 255).astype(np.uint8)