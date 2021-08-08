from gensim.corpora import Dictionary
from gensim.models import TfidfModel


def tfidf_analysis(corpus, doc, rank_number):
    """
    corpus：语料库，形式为二维list，语料库list嵌套句子list，句子list里面是单词
    doc：目标文档，形式同上
    rankNumber：想获取的排名个数,int
    return：当前corpus中，doc中的句子的词的排名rankNumber个数的重要词,list
    """

    dictionary = Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(line) for line in corpus]
    model = TfidfModel(corpus_bow)
    doc_bow = [dictionary.doc2bow(doc[0])]
    vector = model[doc_bow]

    # dictionary.id2token，一直为空
    id2token = {k: v for v, k in dictionary.token2id.items()}
    returnList = []

    for document_number, score in sorted(vector[0], key=lambda x: x[1], reverse=True):
        returnList.append((id2token[document_number], score))

    return returnList[:rank_number]
