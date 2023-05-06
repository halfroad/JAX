import pypinyin

def chinese_to_pinyin(string):
    
    pinyin = pypinyin.pinyin(string, heteronym = True)
    print(pinyin)
    
    pinyin = pypinyin.lazy_pinyin(string)
    print(pinyin)
    
def main():
    
    chinese_to_pinyin("中文")
    
if __name__ == "__main__":
    
    main()
