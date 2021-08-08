import jieba
import string
from tqdm import tqdm


def tokenize_jieba_ch(text, filterStopWords=True):
    """
        text：二维list，每一个list里面一个句子string
        filterStopWords：是否过滤停用词，默认进行过滤
        return：二维list，每一个list里面是词的list
    """

    if filterStopWords:
        stopWords = [' ', ' ', '…']
        table = str.maketrans(string.punctuation, " " * len(string.punctuation))
        rf = open("./stopwords_ch_baidu.txt", "r", encoding="utf-8")
        for line in rf:
            stopWords.append(line.strip('\n'))

    returnList = []
    for sen in tqdm(text, total=len(text)):
        sen = sen.translate(table)
        returnWordList = []
        for word in jieba.cut(sen):
            if filterStopWords and word not in stopWords:
                returnWordList.append(word)
            if not filterStopWords:
                returnWordList.append(word)
        returnList.append(returnWordList)

    return returnList
