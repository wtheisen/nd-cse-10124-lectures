def get_file_as_string(source_file): # Define a function named 'get_lines_str' that takes an argument called 'source_file'
    text = '' # Create a variable named 'lines' points to a list data structure
    with open(source_file) as f:
        for line in f: # For each item in the variable 'f', set the variable named 'line' equal to it, one by one
            text += line.strip() # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)

    return text # This function returns the variable named 'lines'

def get_file_as_list(source_file, special_tokens=False): # Define a function named 'get_lines_str' that takes an argument called 'source_file'
    lines = [] # Create a variable named 'lines' points to a list data structure
    with open(source_file) as f:
        for line in f: # For each item in the variable 'f', set the variable named 'line' equal to it, one by one
            if special_tokens:
                lines.append(['<|sos|>'] + line.strip().split() + ['<|eos|>']) # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)
            else:
                lines.append(line.strip().split()) # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)

    return lines # This function returns the variable named 'lines'

def get_file_as_list_strs(source_file, special_tokens=False): # Define a function named 'get_lines_str' that takes an argument called 'source_file'
    lines = [] # Create a variable named 'lines' points to a list data structure
    with open(source_file) as f:
        for line in f: # For each item in the variable 'f', set the variable named 'line' equal to it, one by one
            if special_tokens:
                lines.append('<|sos|>' + line.strip() + '<|eos|>') # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)
            else:
                lines.append(line.strip()) # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)

    return lines # This function returns the variable named 'lines'

from collections import defaultdict, Counter

def build_graph_word(source_file, file=True, graph=None, special_tokens=True):
    lines = source_file

    if file:
        lines = get_file_as_list(source_file, special_tokens)

    if not graph:
        graph = defaultdict(Counter) # graph is a dictionary of dictionaries like: {'<SOS>': {'I': 37, 'The': 64}}

    for line in lines:
        if line:
            for curr_token, next_token in zip(line, line[1:]):
                graph[curr_token][next_token] += 1

    return graph

def build_graph_char(source_file, graph=None):
    lines = get_file_as_list_strs(source_file)

    if not graph:
        graph = defaultdict(Counter)

    for line in lines:
        if line:
            graph['<|sos|>'][line[0]] += 1

            for idx in range(0, len(line) - 1):
                curr_token = line[idx]
                next_token = line[idx + 1]

                graph[curr_token][next_token] += 1

            graph[line[-1]]['<|eos|>'] += 1

    return graph

def build_graph_token(file_name, tokenizer, vocab_size=512):
    training_str = get_file_as_string(file_name)
    testing_lines = get_file_as_list_strs(file_name, special_tokens=True)

    tokenizer.train(training_str, vocab_size)

    tokenized_lines = []
    for line in testing_lines:
        tokenized_str = tokenizer.encode(line.lower())
        tokenized_lines.append([tokenizer.decode([token]) for token in tokenized_str])

    return build_graph_word(tokenized_lines, file=False)

import random

def generate_sequence(graph, prompt=None, max_token_length=50):
    output = ['<|sos|>']

    while output[-1] != '<|eos|>':
        token_neighbors = graph[output[-1]]

        try:
            output += random.choices(list(token_neighbors.keys()), weights=list(token_neighbors.values()), k=1)
        except:
            output += ['<|eos|>']
            break

        if len(output) > max_token_length:
            break

    return output

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class IrishChatDataset(Dataset):
    def __init__(self, training_file, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.train(get_file_as_string(training_file), 512)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.data = []
        for x in get_file_as_list_strs(training_file, special_tokens=True):
            tokens = torch.tensor(self.tokenizer.encode(x), dtype=torch.long)
            if len(tokens) >= 2:
                # X = tokens[:-1], Y = tokens[1:]  (next-token prediction)
                self.data.append((tokens[:-1], tokens[1:]))
        self.padding_idx = 256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # ---------- Collate (pad, optional one-hot, move to device) ----------
    def collate(self, batch):
        X_list, Y_list = zip(*batch)
        X = pad_sequence(X_list, batch_first=True, padding_value=self.padding_idx)      # (B,T)
        Y_idx = pad_sequence(Y_list, batch_first=True, padding_value=self.padding_idx)  # (B,T)

        V = len(self.tokenizer.vocab)
        Y = F.one_hot(Y_idx.clamp_min(0), num_classes=V).float()                        # (B,T,V)

        # mask True where this position is a real target token (not padding)
        mask = (Y_idx != self.padding_idx)                                              # (B,T)

        return X.to(self.device), Y.to(self.device), mask.to(self.device)
