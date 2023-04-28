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

def purify(string: str, pattern: str = r"[^a-z0-9]", replacement: str = " ", stops = stop_words()):
    
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
        
        labels = []
        titles = []
        descriptions = []
        
        trains = csv.reader(handler)
        
        for line in trains:
            
            labels.append(jax.numpy.float32(line[0]))
            titles.append(purify(line[1]))
            descriptions.append(purify(line[2]))
            
        return labels, titles, descriptions

    
def main():
    
    labels, titles, descriptions = setup()
    
    print(labels[: 5], titles[: 5], titles[: 5])
        
if __name__ == "__main__":
    
    main()
