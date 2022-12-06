import re


def clean_punctuation_marks(string: str):

    return re.sub(r"[^a-z0-9]", repl = "", string = string)

def text_purify(string: str):

    # Lower the string
    string = string.lower()

    # Replace the nonstandard alphabat, includes punctuation makes
    string = re.sub(r"[^a-z0-9]", repl = " ", string = string)

    # Replace multiple spaces
    string = re.sub(r" +", repl = " ", string = string)

    # Trim the spaces in head and tail
    string = string.strip()

    # Split the string with Space
    strings = string.split(" ")

    return strings
    
