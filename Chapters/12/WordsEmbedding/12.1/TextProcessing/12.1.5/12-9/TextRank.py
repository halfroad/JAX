class TextRankScore:

    def __init__(self, strings):

        self.strings = strings
        self.filters = self.__get_strings()
        self.win = self.__get_win()
        self.dictionary = self.__get_text_rank_score_dictionary()

    def __get_strings(self):

        sentences = []

        for string in self.strings:
            for word in string:
                sentences.append(word)

        return sentences

    def __get_win(self):

        win = {}

        for i in range(len(self.filters)):

            if self.filters[i] not in win.keys():
                win[self.filters[i]] = set()

            if i - 5 < 0:
                index = 0
            else:
                index = i - 5

            for j in self.filters[index: i + 5]:
                win[self.filters[i]].add(j)

        return win

    def __get_text_rank_score_dictionary(self):

        time = 0

        score = {w: 1.0 for w in self.filters}

        while time < 50:

            for k, v in self.win.items():

                score_ = score[k] / len(v)
                score[k] = 0

                for i in v:
                    score[i] += score_

            time += 1

        strings_dictionary = {}

        for key in score:
            strings_dictionary[key] = score[key]

        return strings_dictionary

    def __get_text_rank_score(self, string):

        _dict = {}

        for word in string:

            if word in self.dictionary.keys():
                _dict[word] = self.dictionary[word]

        values = sorted(_dict.items(), key = lambda tfidf: tfidf[1], reverse = False)

        _values = []

        for value in values:
            _values.append(value[0])

        return _values
