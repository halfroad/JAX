import re
import jieba

from gensim.models import word2vec
from pypinyin import pinyin, lazy_pinyin, Style

def chineses_2_pinyin(chinese_characters):

    # Ignore the polyphonic charactors
    pinyin_ = lazy_pinyin(chinese_characters)

    return pinyin_

def clean_non_standards(text):

    # Replacce the non-standard characters, ^ is to reverse
    text = re.sub(r"[a-zA-Z0-9-，。“”、：《》（）()]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(" ", "", text)

    return text

def cut_chinese_sentences(chinese_sentences):

    array = jieba.lcut_for_search(chinese_sentences)

    return array

def vectorize_chinese_characters(chinese_characters_array):

    model = word2vec.Word2Vec(chinese_characters_array, vector_size = 50, min_count = 1, window = 3)

    return model

def start():

    text = "您好"

    pinyin_ = chineses_2_pinyin(text)

    print(pinyin_)

    text = "中新社联合国10月18日电 中国裁军大使李松17日在第77届联合国大会裁军与国际安全委员会(联大一委)专题讨论中发言，就核裁军问题提出中方七点主张。李松说，中国一贯主张全面禁止和彻底销毁核武器，最终实现无核武器世界目标。《不扩散核武器条约》(NPT)无限期延长，不意味着核武器国家得以永远拥有核武器。李松说，当前，全球战略安全环境持续恶化，霸权主义、强权政治，冷战思维、意识形态划线，大国竞争、阵营对抗这样的理念和政策，严重威胁国际和平与安全。核武器作用、核战争风险等问题再度引起国际社会高度关注。核裁军何去何从，联合国需要答案。中国主张：一、国际社会应践行真正的多边主义，秉持共同、综合、合作、可持续的安全观。大国，特别是核武器国家，必须摒弃战略竞争、以意识形态划线、阵营对立对抗理念，放下独享安全、绝对安全的执念，不将本国安全凌驾于他国安全之上，不利用核武器称王争霸、欺凌胁迫无核武器国家。二、美国和俄罗斯作为依然拥有最庞大核武库的核超级大国，应继续履行核裁军特殊、优先历史责任，以可核查、不可逆、具有法律约束力的方式，进一步大幅、实质削减各自的核武库，为最终实现全面、彻底核裁军创造条件。鉴于核武器国家在核政策、核力量、安全环境等方面存在巨大差异，核军控、核裁减、核透明不存在统一模板，须遵循“维护全球战略稳定”“各国安全不受减损”等原则，以公平合理、逐步削减、向下平衡的方式，循序渐进推进核裁军进程。三、核武器国家应对有关核战略与核政策作出切实调整，降低核武器在国家安全政策中的作用，承诺不首先使用核武器，不把任何国家列为核打击目标，不将核武器瞄准任何国家，无条件不对无核武器国家和无核武器区使用或威胁使用核武器。中国呼吁五核国缔结“互不首先使用核武器条约”，并积极推动日内瓦裁谈会谈判缔结对无核武器国家提供“消极安全保证”的国际法律文书。四、“核共享”与NPT宗旨和原则背道而驰，不应鼓励、不得扩散。与核武器国家结盟的无核武器国家，与其他无核武器国家具有很重要的不同之处，安全诉求也并不完全一致。这些国家也有必要承担责任、作出努力，切实降低核武器在其国家安全战略和集体安全战略中的作用。五、今年1月，五核国领导人发表《关于防止核战争与避免军备竞赛的联合声明》，申明“核战争打不赢也打不得”。这一历史性声明及时发表，对防止核战争、维护全球战略稳定具有重要深远意义，必须得到严肃认真的恪守。五核国应进一步就战略稳定、减少核风险等问题加强沟通，并可着手围绕反导、外空、网络、人工智能等更广泛议题开展深入对话，重建互信，加强合作。六、必须坚决抵制损害国际核不扩散体系的错误做法。个别核武器国家把地缘政治利益凌驾于核不扩散目标之上，奉行双重标准和实用主义，与无核武器国家开展违背NPT目的和宗旨的核潜艇合作，并企图在亚太地区复制“核共享”。国际社会应旗帜鲜明地反对上述核扩散行径，共同创造有利于核裁军取得进展的国际和地区安全环境。七、NPT缔约国应以新一轮审议周期为契机，坚定维护条约权威性和有效性，进一步加强国际核不扩散机制，推动NPT服务和平与发展。努力推动《全面禁止核试验条约》早日生效，加强履约准备工作，恪守“暂停试”承诺。支持日内瓦裁谈会在全面平衡工作计划基础上，根据“香农报告”所载授权启动“禁止生产核武器用裂变材料条约”谈判，通过具有法律约束力的方式实现禁产目标。"

    text = clean_non_standards(text)

    print(text)

    array = cut_chinese_sentences(text)

    print(array)

    model = vectorize_chinese_characters([array])

    print(model.wv["核武器"])

if __name__ == "__main__":

    start()
