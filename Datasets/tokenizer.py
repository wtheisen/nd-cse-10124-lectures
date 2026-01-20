"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

class Simple_Tokenizer():
    def __init__(self):
        self.merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        self.vocab = vocab | {256: b'<|sos|>', 257: b'<|eos|>'}


    def _get_stats(self, ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts

        for pair in zip(ids, ids[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def _merge(self, ids, pair, idx):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        newids = []
        i = 0

        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1

        return newids

    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 258

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = self._get_stats(ids)

            # find the pair with the highest count
            pair = max(stats, key=stats.get)

            # mint a new token: assign it the next available id
            idx = 258 + i

            # replace all occurrences of pair in ids with idx
            ids = self._merge(ids, pair, idx)

            # save the merge
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore

            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)

        return ids

    def visualize_tokenization(self, ids):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'

        tokens = []
        for token_id in ids:
            token_str = self.decode([token_id])
            tokens.append(f"{GREEN}{token_str}{GRAY}({token_id}){RESET}")

            if token_str in ['<|SOS|>', '<|EOS|>']:
                tokens.append('\n\n\t')

        return ' | '.join(tokens)