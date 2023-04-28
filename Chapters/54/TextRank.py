class TextRank:
    
    def __init__(self, strings):
        
        self.strings = strings
        self.filters = self.__get_words()
        self.win = self.__get_win()
        self.dictionary = self.__get_dictionary()
        
    def __get_words(self):
        
        words = []
        
        for string in self.strings:
            
            for word in string:
                
                words.append(word)
                
        return words;
    
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
    
    def __get_dictionary(self):
        
        time = 0
        
        scores = {w: 1.0 for w in self.filters}
        
        while time < 50:
            
            for key, value in self.win.items():
                
                score = scores[key] / len(value)
                
                scores[k] = 0
                
                for i in value:
                    
                    scores[i] += score
                    
            time += 1
            
        _dictionary = {}
        
        for key in scores:
            
            self.dictionary[key] = scores[key]
            
        return self.dictionary
    
    def __get_text_rank(self, string):
        
        _dictionary = {}
        values = []
        
        for word in string:
            
            if word in self.dictionary.keys():
                
                _dictionary[word] = self.dictionary[word]
                
        sorted_values = sorted(_dictionary.items(), key = lambda word_tfidf: word_tfidf[1], reverse = False)
        
        for value in sorted_values:
            
            values.append(value[0])
            
        return values
    
    