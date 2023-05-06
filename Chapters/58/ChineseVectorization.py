def vectorize():
    
    string = "近日，哈尔滨市松北区多名小区业主报料称，一个健身房老板拆穿了承重墙，造成小区整栋居民楼的住户被疏散。该房屋为高层住宅，一共31层，大部分业主都是通过贷款买的房子，而健身房老板拆承重墙的行为，导致200多户居民无家可归。消息一经披露，迅速引发公众关注和热议。“施工队在3楼给承重墙拆了，已经出现墙体开裂了，闹得业主都人心惶惶。”“普通镐子打不动，这家伙直接上专业的拆迁钩机。”“从建筑学的角度来讲，这楼废了。应力已经重新分配了，结构体系转变了，补回去没用了……这是危害公共安全！”事发地的居民楼，是松北区裕民街道利民学苑B栋2单元。4月28日当天凌晨一点多，居民楼里240多户业主被紧急疏散。目前，业主们被临时安置到附近的酒店宾馆，当地住建部门和警方已经介入调查。据悉，涉事的居民楼下，周围有宾馆、网吧、火锅店多家商铺。当地人士称，“那天经过这里，还觉得奇怪，几种大型机器齐上阵，最后把居民楼搞废了。”“有人租了居民楼的三楼，准备开健身房，装修时把承重墙砸穿了。事发当天，租户就被警方控制了。”"
    
    import jieba
    import re
    
    string = re.sub(r"[a-zA-Z0-9-，。""”“！、（）《》【】…]", " ", string)
    string = re.sub(r" +", " ", string)
    string = re.sub(" ", "", string)
    
    print(string)
    
    strings = jieba.lcut_for_search(string)
    
    print(strings)
    
    import gensim.models
    
    model = gensim.models.word2vec.Word2Vec([strings], vector_size = 50, min_count = 1, window = 3)
    
    print(model.wv["小区业主"])
    
def main():
    
    vectorize()
    
if __name__ == "__main__":
    
    main()


    
    
