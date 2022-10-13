from InverseDocumentFrequency import compute_inverse_document_frequency

def compute_term_and_inverse_document_frequency(texts):

    inverseDocumentFrequencys = compute_inverse_document_frequency(texts)

    for text in texts:

        term_and_inverse_document_frequency = {}

        for word in text:

            if word in term_and_inverse_document_frequency:
                term_and_inverse_document_frequency[word] += 1
            else:
                term_and_inverse_document_frequency[word] = 1

        for word in term_and_inverse_document_frequency:

            term_and_inverse_document_frequency[word] *= inverseDocumentFrequencys[word]
