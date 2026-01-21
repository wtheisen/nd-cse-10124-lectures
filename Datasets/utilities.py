def get_file_as_string(source_file): # Define a function named 'get_lines_str' that takes an argument called 'source_file'
    text = '' # Create a variable named 'lines' points to a list data structure
    with open(source_file) as f:
        for line in f: # For each item in the variable 'f', set the variable named 'line' equal to it, one by one
            text += line.strip() # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)

    return text # This function returns the variable named 'lines'

def get_file_as_list(source_file): # Define a function named 'get_lines_str' that takes an argument called 'source_file'
    lines = [] # Create a variable named 'lines' points to a list data structure
    with open(source_file) as f:
        for line in f: # For each item in the variable 'f', set the variable named 'line' equal to it, one by one
            lines.append(line.strip().split()) # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)

    return lines # This function returns the variable named 'lines'

def get_file_as_list_strs(source_file): # Define a function named 'get_lines_str' that takes an argument called 'source_file'
    lines = [] # Create a variable named 'lines' points to a list data structure
    with open(source_file) as f:
        for line in f: # For each item in the variable 'f', set the variable named 'line' equal to it, one by one
            lines.append(line.strip()) # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)

    return lines # This function returns the variable named 'lines'

from collections import defaultdict, Counter

def build_graph_word(source_file, file=True, graph=None):
    lines = source_file

    if file:
        lines = get_file_as_list(source_file)

    if not graph:
        graph = defaultdict(Counter) # graph is a dictionary of dictionaries like: {'<SOS>': {'I': 37, 'The': 64}}

    for line in lines:
        if line:
            graph['<|sos|>'][line[0]] += 1

            for idx in range(0, len(line) - 1):
                curr_token = line[idx]
                next_token = line[idx + 1]

                graph[curr_token][next_token] += 1

            graph[line[-1]]['<|eos|>'] += 1

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

def create_token_graph(file_name, tokenizer, vocab_size=512):
    training_str = get_file_as_string(file_name)
    testing_lines = get_file_as_list_strs(file_name)

    tokenizer.train(training_str, vocab_size)

    tokenized_lines = []
    for line in testing_lines:
        tokenized_str = tokenizer.encode(line.lower())
        tokenized_lines.append([tokenizer.decode([token]) for token in tokenized_str])

    token_graph = build_graph_word(tokenized_lines, file=False)

    return token_graph