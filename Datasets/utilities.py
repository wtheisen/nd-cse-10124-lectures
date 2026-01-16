def get_file_str(source_file): # Define a function named 'get_lines_str' that takes an argument called 'source_file'
    text = '' # Create a variable named 'lines' points to a list data structure
    with open(source_file) as f:
        for line in f: # For each item in the variable 'f', set the variable named 'line' equal to it, one by one
            text += '<SOS> ' + line.strip() + ' <EOS>' # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)

    return text # This function returns the variable named 'lines'

def get_lines_str(source_file): # Define a function named 'get_lines_str' that takes an argument called 'source_file'
    lines = [] # Create a variable named 'lines' points to a list data structure
    with open(source_file) as f:
        for line in f: # For each item in the variable 'f', set the variable named 'line' equal to it, one by one
            lines.append('<SOS> ' + line.strip() + ' <EOS>') # Add the content of the variable 'line' to the end of the list named 'lines' (after removing whitespace and newlines on either end)

    return lines # This function returns the variable named 'lines'