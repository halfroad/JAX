import re

import jieba
import pypinyin
from gensim.models import word2vec


def greet():

    ninhao = pypinyin.lazy_pinyin("您好")

    print(ninhao)

def purify(string):

    string = re.sub(r"[a-zA-Z0-9-，、。“”（）－()-,.""]", " ", string)
    string = re.sub(r" +", " ", string)
    string = re.sub(" ", "", string)

    return string

def vectorize(string, characters):

    words = jieba.lcut(string)
    model = word2vec.Word2Vec([words], vector_size = 50, min_count = 1, window = 5)

    vectors = model.wv[characters]

    return vectors

def test():

    string = "2月10日，习近平主席结束出席首届中国－阿拉伯国家峰会(Sino-Arab Submmit)、中国－海湾阿拉伯国家合作委员会峰会并对沙特进行国事访问回到国内。这次新中国成立以来我国对阿拉伯世界规模最大、规格最高的外交行动，必将以其划时代的里程碑意义在中阿关系发展史上留下光辉灿烂的一页，必将对国际格局和地区形势产生深远影响。中阿友好是一部绵延千年的交往史、更是一部伟大精神的传承史。从丝绸古道商旅络绎，到上世纪双方全面建交，再到新世纪中阿论坛开启整体合作时代，共建“一带一路”形成合作新格局，中阿友好如涓涓细流汇聚成尼罗河水般奔涌向前，蓬勃生机的背后蕴含着人类文明友好交往的“精神密码”。"
    string = purify(string)

    vectors = vectorize(string, "习近平")

    print(vectors)

if __name__ == '__main__':

    test()
