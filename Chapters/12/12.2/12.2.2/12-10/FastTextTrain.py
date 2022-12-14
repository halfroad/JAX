import jieba as jieba

from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec


def setup():

    sentences = [
        "卷积神经网络在图像处理领域获得了极大成功，其结合特征提取和目标训练为一体的模型能够最好地利用已有的信息对结果进行反馈训练。",
        "对于文本识别的卷积神经网络来说，同样也是充分利用特征提取时提取的文本特征来计算文本特征权值大小的，归一化处理需要处理的数据。",
        "这样使得原来的文本信息抽象成一个向量化的样本集，之后将样本集和训练好的模板输入卷积神经网络进行处理。",
        "本节将在上一节的基础上使用卷积神经网络实现文本分类的问题，这里将采用两种主要基于字符的和基于词嵌入形式的词卷积神经网络处理方法。",
        "实际上无论是基于字符的还是基于词嵌入形式的处理方式都是可以相互转换的，这里只介绍使用基本的使用模型和方法，更多的应用还需要读者自行挖掘和设计。"
    ]

    return sentences

def sentence_cut(sentences):

    cuts = []

    for sentence in sentences:

        cut = jieba.lcut(sentence)

        cuts.append(cut)

    return cuts

class TrainProgressHandler(CallbackAny2Vec):

    """

    Output loss at each epoch

    """

    def __init__(self):

        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model_):

        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model_):

        loss = model_.get_latest_training_loss()

        self.losses.append(loss)

        print(f'  Loss: {loss}')

        self.epoch += 1

def train(cuts, path):

    handler = TrainProgressHandler()

    model = FastText(vector_size = 4, window = 3, min_count = 1, sentences = cuts, epochs = 10, callbacks = [handler])

    model.build_vocab(cuts)
    model.train(cuts, total_examples = model.corpus_count, epochs = 10, callbacks = [handler])

    model.save(path)

def acquire_parameters(path):

    model = FastText.load(path)

    print("model.wv.key_to_index =", model.wv.key_to_index)
    print("model.wv.index_to_key =", model.wv.index_to_key)
    print("model.wv.vectors[: 3] =", model.wv.vectors[: 3])
    print("len(model.wv.vectors) =", len(model.wv.vectors))
    print("len(model.wv.index_to_key) =", len(model.wv.index_to_key))

    embedding = model.wv["卷积", "神经网络"]

    print("embedding =", embedding)

def start():

    sentences = setup()

    cuts = sentence_cut(sentences)

    print(cuts)

    path = "../../../../../Exclusion/Models/FastText/FastText_JieBa_Model.model"

    train(cuts, path)

    acquire_parameters(path)

def main():

    start()

if __name__ == "__main__":

    main()
