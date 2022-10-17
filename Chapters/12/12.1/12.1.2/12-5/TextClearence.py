import re

from nltk import PorterStemmer


def text_clear(text, stops, allow_prefix_suffix = True, split = False, stem = False):

    text = text.lower()

    # ^ is to reverse, to replace the non-standard characters
    text = re.sub(r"[^a-z]", " ", text)
    # Remove multiple SPACEs
    text = re.sub(r" +", " ", text)
    # Remove the SPACEs from leading and trailing

    text = text.strip()

    if split:
        text = text.split(" ")

    if stem:
        text = [word for word in text if word not in stops]
        text = [PorterStemmer().stem(word) for word in text]

    if allow_prefix_suffix:

        text = text.join("eos")
        # text = ["bos"] + text

    return text
