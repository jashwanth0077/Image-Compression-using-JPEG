import numpy as np
import sys
from collections import Counter, defaultdict
import heapq

def run_length_encode(block):
    """Run-length encoding of an 8x8 block (zigzag scan assumed)."""
    flat = zigzag_scan(block)
    rle = []
    run = 0

    for value in flat:
        if value == 0:
            run += 1
        else:
            size = int(np.ceil(np.log2(abs(value) + 1)))
            rle.append((run, size, value))
            run = 0

    return rle

def zigzag_scan(block):
    """Zigzag scan of an 8x8 block."""
    indices = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]
    return [block[i, j] for i, j in indices]

def zigzag_to_block(flat_array, block_size=8):
    """
    Reconstruct an 8x8 block from a zigzag-scanned 1D array.

    Args:
        flat_array (list or ndarray): 1D array in zigzag order.
        block_size (int): Size of the block (default: 8x8).

    Returns:
        ndarray: 2D array reconstructed in standard block format.
    """

    # Define the zigzag order indices
    indices = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    # Initialize an empty 2D block
    block = np.zeros((block_size, block_size), dtype=np.float32)

    # Fill the block using the zigzag order
    for idx, (i, j) in enumerate(indices):
        block[i, j] = flat_array[idx] if idx < len(flat_array) else 0

    return block


def huffman_encode_rle_with_global_codes(rle, codes):
    """
    Encode the RLE output using global Huffman codes.
    - `run` and `size` are directly represented in 4 bits each.
    - `value` is Huffman-encoded using global Huffman codes.
    """
    encoded_rle = []
    for run, size, value in rle:
        run_bits = f"{run:04b}"  # 4 bits for run
        value_code = codes[value]
        size_bits = f"{len(value_code):04b}"  # 4 bits for size
        # print(run_bits + size_bits + value_code)
        encoded_rle.append(run_bits + size_bits + value_code)

    return "".join(encoded_rle)

class HuffmanCoding:
    def build_tree(self, freq):
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        return dict(heap[0][1:])

    def encode(self, symbols):
        freq = Counter(symbols)
        self.codes = self.build_tree(freq)
        return self.codes

    def decode(self, encoded_data):
        """
        Decode a binary string encoded with global Huffman codes.
        Format:
        - First 4 bits: Number of zeros (run-length) before the Huffman code.
        - Next 4 bits: Size of the Huffman code (number of bits in the code).
        - Next `size` bits: The actual Huffman code.

        Args:
            encoded_data (str): The encoded binary string.

        Returns:
            list: A 1D array representing the reconstructed sequence, including zeros.
        """
        reverse_codes = {v: k for k, v in self.codes.items()}  # Reverse mapping for decoding
        decoded = []  # Final result array
        index = 0  # Pointer in the encoded binary string
        j = 0
        initial_run_length = 0
        while index+8 < len(encoded_data):
            # Read the run-length (4 bits)
            run_length = int(encoded_data[index:index + 4], 2)
            index += 4
            # print(run_length)
            # Read the size of the Huffman code (4 bits)
            size = int(encoded_data[index:index + 4], 2)
            index += 4

            # Read the Huffman code of the specified size
            huffman_code = encoded_data[index:index + size]
            index += size

            # Decode the Huffman code to find the value
            value = reverse_codes.get(huffman_code, None)
            if value is None:
                # print(encoded_data[index-size-8:index])
                # print(decoded)
                # raise ValueError(f"Invalid Huffman code: {huffman_code}")
                value = 0

            # Append the zeros (based on run-length) and the value to the decoded array
            decoded.extend([0] * (run_length-initial_run_length))
            decoded.append(value)
            j += 1
            # print(value)
            initial_run_length = run_length
        return decoded