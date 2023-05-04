import jieba
import gensim.models
import os

def setup():
    
    sentences = ["泽连斯基否认袭击克里姆林宫：我们甚至没有足够的武器",
              "“我们没有攻击普京或莫斯科”。据《基辅独立报》报道，乌克兰总统泽连斯基当地时间5月3日公开否认克里姆林宫遭乌克兰无人机袭击的说法。",
              "“我们没有攻击普京或莫斯科。我们在自己的领土上战斗，保卫我们的村庄和城市。”泽连斯基在芬兰参加为期一天的北欧国家领导人峰会期间，在新闻发布会上回应克宫遭袭时称，“我们甚至没有足够的武器。”",
              "“因此，我们没有攻击普京。我们将把其留给法庭。”泽连斯基称。",
              "报道说，3日早些时候，泽连斯基的发言人在向英国广播公司（BBC）乌克兰频道发表的声明中也否认了乌方参与“袭击克里姆林宫”。"
              ]
    cuts = []
    
    for sentence in sentences:
        
        cut = jieba.lcut(sentence)
        cuts.append(cut)
        
    return cuts

def train(path):
    
    cuts = setup()
    
    if os.path.exists(path):
        
        model = gensim.models.FastText.load(path)
        
        return model
    
    else:
        
        model = gensim.models.FastText(vector_size = 4, window = 3, min_count = 1, sentences = cuts, epochs = 10)
    
        model.build_vocab(cuts)
        model.train(cuts, total_examples = model.corpus_count, epochs = 10)
    
        model.save(path)
        
        return model

def main():
    
    path = "/tmp/FastText_Jieba_Model.model"
    
    model = train(path)
    
    print("model.wv.key_to_index = ", model.wv.key_to_index)
    print("model.wv.index_to_key = ", model.wv.index_to_key)
    print("len(model.wv.vectors) = ", len(model.wv.vectors))
    
    
    embedding = model.wv["克里姆林宫", "我们"]
    
    print("embedding = ", embedding)
        
    
if __name__ == "__main__":
    
    main()
    
