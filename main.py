from tfidf import tfidf_analysis
from load_data import table_data_filter
from my_tokenize import tokenize_jieba_ch
from tqdm import tqdm

data_corpus = table_data_filter("./data.csv", name_ch='疯狂动物城')
data_documents = table_data_filter("./data.csv", name_ch='疯狂动物城', star="5")

corpus = []
for d in tqdm(data_corpus, total=len(data_corpus)):
    corpus.append(d["comment"])
documents = ""
for d in tqdm(data_documents, total=len(data_documents)):
    documents += d["comment"]
    documents += " "

corpus = tokenize_jieba_ch(corpus)
documents = tokenize_jieba_ch([documents])

results = tfidf_analysis(corpus=corpus, doc=documents, rank_number=30)
for r in results:
    print(r)
