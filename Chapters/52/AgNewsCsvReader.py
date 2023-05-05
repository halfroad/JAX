import csv
import re
import jax
import ssl
import nltk
    
def stop_words():
    
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.data.path.append("/tmp/")
    
    nltk.download("stopwords", download_dir = "/tmp/");
    
    stops = nltk.corpus.stopwords.words("english")
    
    print(stops)
    
    return stops

def purify(string: str, pattern: str = r"[^a-z]", replacement: str = " "):
    
    string = string.lower()
    
    string = re.sub(pattern = pattern, repl = replacement, string = string)
    # Replace the consucutive spaces with single space
    string = re.sub(pattern = r" +",  repl = replacement, string = string)
    # string = re.sub(pattern = " ", repl = "", string = string)
    
    # Trim the string
    string = string.strip()
    string = string + " eos"
    
    return string

def purify_stops(string: str, pattern: str = r"[^a-z0-9]", replacement: str = " ", stops = stop_words()):
    
    string = string.lower()
    
    string = re.sub(pattern = pattern, repl = replacement, string = string)
    # Replace the consucutive spaces with single space
    string = re.sub(pattern = r" +",  repl = replacement, string = string)
    
    # Trim the string
    string = string.strip()
    
    # Seperate the string with space, an array will be yielded
    strings = string.split(" ")
    
    strings = [word for word in strings if word not in stops]
    strings = [nltk.PorterStemmer().stem(word) for word in strings]
    
    strings.append("eos")
    strings = ["bos"] + strings
    
    return strings

def setup():
    
    with open("../../Shares/ag_news_csv/train.csv", "r") as handler:
        
        train_labels = []
        train_titles = []
        train_descriptions = []
        
        trains = csv.reader(handler)
        trains = list(trains)
        
        for i in range(len(trains)):
            
            line = trains[i]
            
            train_labels.append(jax.numpy.int32(line[0]))
            train_titles.append(purify(line[1]))
            train_descriptions.append(purify_stops(line[2]))
            
    with open("../../Shares/ag_news_csv/test.csv", "r") as handler:
        
        test_labels = []
        test_titles = []
        test_descriptions = []
        
        tests = csv.reader(handler)
        tests = list(tests)
        
        for i in range(len(tests)):
            
            line = tests[i]
            
            test_labels.append(jax.numpy.int32(line[0]))
            test_titles.append(purify(line[1]))
            test_descriptions.append(purify_stops(line[2]))
            
        return (train_labels, train_titles, train_descriptions), (test_labels, test_titles, test_descriptions)

    
def main():
    
    (train_labels, train_titles, train_descriptions), (test_labels, test_titles, test_descriptions) = setup()
    
    print((train_labels.shape, train_titles.shape, train_descriptions.shape), (test_labels.shape, test_titles.shape, test_descriptions.shape))
        
if __name__ == "__main__":
    
    main()
