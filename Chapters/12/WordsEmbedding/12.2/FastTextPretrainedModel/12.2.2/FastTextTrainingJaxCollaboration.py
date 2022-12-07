import gensim
import jieba


def split(sentences):

    phrases = []

    for sentence in sentences:

        phrase = jieba.lcut(sentence)
        phrases.append(phrase)

    return phrases

def fast_text(sentences):

    model = gensim.models.FastText(vector_size = 4, window = 2, min_count = 1, sentences = sentences, epochs = 10)


def test():

    sentences = [
        "卷积神经网络在图像处理领域获得了极大成功，其结合特征提取和目标训练为一体的模型能够最好地利用已有的信息对结果进行反馈训练。",
        "对于文本识别的卷积神经网络来说，同样也是充分利用特征提取时提取的文本特征来计算文本特征权值大小的，归一化处理需要处理的数据。",
        "这样使得原来的文本信息抽象成一个向量化的样本集，之后将样本集和训练好的模板输入卷积神经网络进行处理。",
        "本节将在上一节的基础上使用卷积神经网络实现文本分类的问题，这里将采用两种主要基于字符的和基于词嵌入形式的词卷积神经网络处理方法。",
        "实际上无论是基于字符的还是基于词嵌入形式的处理方式都是可以相互转换的，这里只介绍使用基本的使用模型和方法，更多的应用还需要读者自行挖掘和设计。"
    ]

    phrases = split(sentences)

    print(phrases)

if __name__ == '__main__':

    test()



