import csv
import re
import datasets

import sys

sys.path.append("../12.1.2/12-5/")
from TextClearence import text_clear

from datasets import GeneratorBasedBuilder, DatasetInfo, Features, DownloadManager

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

class AGNewsGeneratorBasedBuilder(GeneratorBasedBuilder):

    """

    AG News topic classification dataset.

    https://huggingface.co/datasets/ag_news/blob/main/ag_news.py

    """

    def __init__(self, stops, refine = True, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.stops = stops
        self.refine = refine

    def _info(self) -> DatasetInfo:

        return datasets.DatasetInfo(

            description = "AG News topic classification dataset.",

            features = Features({

                "text": datasets.Value("string"),
                "label": datasets.features.ClassLabel(names = ["World", "Sports", "Business", "Sci/Tech"]),
            }),
            homepage = "http://groups.di.unipi.it",
            citation = "citation",
            task_templates = [datasets.TextClassification(text_column = "text", label_column = "label")]
        )

    def _split_generators(self, dl_manager: DownloadManager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datasets.SplitGenerator(name = datasets.Split.TRAIN, gen_kwargs = {"filepath": train_path}),
            datasets.SplitGenerator(name = datasets.Split.TEST, gen_kwargs = {"filepath": test_path})
        ]

    def _generate_examples(self, **kwargs):

        """

        Generate AG News examples.

        """

        filepath = kwargs["filepath"]

        with open(filepath, encoding = "utf-8") as handle:

            reader = csv.reader(handle)

            for id_, row in enumerate(reader):

                label, title, description = row

                text = " ".join((title, description))

                if self.refine:

                    text = text_clear(text, self.stops)

                # Original labels are [1, 2, 3, 4] ->
                #                     ["World", "Sports", "Business", "Sci/Tech"]
                # Re-map to [0, 1, 2, 3]
                label = int(label) - 1

                yield id_, {"text": text, "label": label}


class AGNewsDatasetAutoGenerator:

    def __init__(self, stops, cache_dir = "../../../../Exclusion/Datasets/AGNews/Cache/"):

        self.stops = stops
        self.cache_dir = cache_dir

    def prepare(self, cache_dir = "../../../../Exclusion/Datasets/AGNews/Cache/"):

        builder = AGNewsGeneratorBasedBuilder(cache_dir = self.cache_dir, stops = self.stops)

        builder.download_and_prepare()

        dataset = builder.as_dataset(split = [datasets.Split.TRAIN, datasets.Split.TEST])

        trains = dataset[0]
        tests = dataset[1]

        return trains, tests




