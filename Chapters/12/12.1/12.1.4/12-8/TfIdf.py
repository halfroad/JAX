import math
import sys

sys.path.append("../../12.1.1/12-1/")
sys.path.append("../../12.1.2/12-5/")
from DatasetsBuilder import read

class TermAndInverseDocumentFrequencyScore:

    def __init__(self, corpus, model = None):

        self.corpus = corpus
        self.model = model
        self.inverse_document_frequencys = self.compute_inverse_document_frequency()

    def compute_inverse_document_frequency(self):

        inverse_document_frequencys = {}
        number_of_documents = 0.0

        for document in self.corpus:

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

    def compute_term_and_inverse_document_frequency_score(self, text):

        word_term_and_inverse_document_frequency = {}

        for word in text:

            if word in word_term_and_inverse_document_frequency:
                word_term_and_inverse_document_frequency[word] += 1
            else:
                word_term_and_inverse_document_frequency[word] = 1

        for word in word_term_and_inverse_document_frequency:

            word_term_and_inverse_document_frequency[word] *= self.inverse_document_frequencys[word]

        values = sorted(word_term_and_inverse_document_frequency.items(), key = lambda item: item[1], reverse = True)
        values = [value[0] for value in values]

        return values


def start():

    root = "../../../../../Exclusion/Datasets/"
    stops, trains, tests = read(root = root)
    texts = trains["description"]

    tfidf = TermAndInverseDocumentFrequencyScore(texts)

    for text in texts:

        values = tfidf.compute_term_and_inverse_document_frequency_score(text)

        print(values)

def main():

    start()

if __name__ == "__main__":

    main()
