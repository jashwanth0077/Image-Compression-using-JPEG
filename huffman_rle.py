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
        reverse_codes = {v: k for k, v in self.codes.items()}
        current_code = ""
        decoded = []

        for bit in encoded_data:
            current_code += bit
            if current_code in reverse_codes:
                decoded.append(reverse_codes[current_code])
                current_code = ""

        return decoded