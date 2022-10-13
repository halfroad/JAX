import math


def compute_inverse_document_frequency(corpus):

    inverse_document_frequencys = {}
    number_of_documents = 0.0

    for document in corpus:

        number_of_documents += 1
        counted = []

        # Count the occurences for the given word
        for word in document:

            if word not in counted:

                counted.append(word)

                if word in inverse_document_frequencys:
                    inverse_document_frequencys[word] += 1
                else:
                    inverse_document_frequencys[word] = 1

    # Compute the Inverse Document Frequency for each word
    for word in inverse_document_frequencys:
        inverse_document_frequencys[word] = math.log(number_of_documents / float(inverse_document_frequencys[word]))

    return inverse_document_frequencys

def compute_term_and_inverse_document_frequency(texts):

    inverse_document_frequencys = compute_inverse_document_frequency(texts)

    for text in texts:

        word_term_and_inverse_document_frequency = {}

        for word in text:

            if word in word_term_and_inverse_document_frequency:
                word_term_and_inverse_document_frequency[word] += 1
            else:
                word_term_and_inverse_document_frequency[word] = 1

        for word in word_term_and_inverse_document_frequency:

            word_term_and_inverse_document_frequency[word] *= inverse_document_frequencys[word]

        values = sorted(word_term_and_inverse_document_frequency.items(), key = lambda item: item[1], reverse = True)
        # values = [value[0] for value in values]

        return values



                
