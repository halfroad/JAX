import ssl
import nltk
from nltk.corpus import stopwords

from AGNews import AGNewsDatasetAutoGenerator

def download_stop_words(path):

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("stopwords", download_dir = path)


def read(root = "../../../../Exclusion/Datasets/", mltk_directory = "MLTK/", agnews_directory = "AGNews/Cache/"):

    path = root + mltk_directory

    download_stop_words(path)
    nltk.data.path.append(path)

    stops = stopwords.words("english")

    print(stops)

    path = root + agnews_directory

    trains, tests = AGNewsDatasetAutoGenerator(stops = stops, cache_dir = path).prepare()

    return stops, trains, tests
