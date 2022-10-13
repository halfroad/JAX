class TextRankScore:

    def __init__(self, texts):

        self.texts = texts
        self.filters = self.get_sentences()
        self.win = self.get_win()
        self.texts_dictionary = self.get_text_rank_score_dictionary()

    def get_sentences(self):

        sentences = []

        for text in self.texts:

            for word in text:

                sentences.append(word)

        return sentences

    def get_win(self):

        win = {}

        for i in range(len(self.filters)):

            win[self.filters[i]] = set()

        if i - 5 < 0:
            index = 0
        else:
            index = i - 5

        for j in self.filters[index: i + 5]:

            win[self.self.filters[i]].add(j)

        return win

    def get_text_rank_score_dictionary(self):

        time = 0
        scores = {w: 1.0 for w in self.filters}

        while time < 50:

            for key, value in self.win.items():

                score = scores[key] / len(value)
                score[key] = 0

                for i in value:

                    scores[i] += score

                time += 1

        texts_dictionary = {}

        for key in scores:

            texts_dictionary[key] = scores[key]

        return texts_dictionary

    def get_text_rank_score(self, text):

        _dict = {}

        for word in text:

            if word in self.texts_dictionary.keys():

                _dict[word] = self.texts_dictionary[word]

        values = sorted(_dict.items(), key = lambda tfidf: tfidf[1], reverse = False)

        return values

    def get_text_rank_result(self, text):

        _dict = {}

        for word in text:

            if word in self.texts_dictionary.keys():

                _dict[word] = self.texts_dictionary[word]

            values = sorted(_dict.items(), key = lambda tfidf: tfidf[1], reverse = False)
            values_ = []

            for value in values:

                values_.append(value[0])

            return values_




