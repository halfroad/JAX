import math


def compute_inverse_document_frequency(corpus):

    """

    Parameters,
        corpus: the number of all the input documents

    """

    inverseDocumentFrequencys = {}

    occurences = 0.0

    # Count occurences of the given word
    for document in corpus:

        occurences += 1
        counted = []

        for word in document:

            if word not in counted:

                counted.append(word)

                if word in inverseDocumentFrequencys:
                    inverseDocumentFrequencys[word] += 1
                else:
                    inverseDocumentFrequencys[word] = 1

    # Compute the Inverse Document Frequency for the given word
    for word in inverseDocumentFrequencys:

        inverseDocumentFrequencys[word] = math.log(occurences / float(inverseDocumentFrequencys[word]))

    return inverseDocumentFrequencys

