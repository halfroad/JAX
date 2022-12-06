import math


def inverse_document_frequency(corpus):

    """
    Corpus: All the Corporas
    """

    idfs = {}
    d = 0.0

    # Statistics of occurences of words
    for document in corpus:

        d += 1
        counted = []

        for word in document:

            if word not in counted:

                counted.append(word)

                if word in idfs:
                    idfs[word] += 1
                else:
                    idfs[word] = 1

    # Compute the Inverse Document Frequency

    for word in idfs:

        idfs[word] = math.log(d / float(idfs[word]))

    return idfs

def compute_idfs(strings):

    """

    Compute the IDF for each word in strings, the strings is the processed corpus

    """

    idfs = inverse_document_frequency(strings)

    for string in strings:

        # TF-IDF for each word
        tfidf = {}

        for word in string:

            if word in tfidf:
                tfidf[word] += 1
            else:
                tfidf[word] = 1

        for word in tfidf:
            
            tfidf[word] *= idfs[word]

        # Sort by value
        values = sorted(tfidf.items(), key = lambda item: item[1], reverse = True)
        values = [value[0] for value in values]

        return values
