import json
import csv
import string
from gensim.models import Word2Vec

"""
需求：将json中的数据转换成csv文件
"""
def txt_json():

    json_fp = open("D://data//Paper recommendation system//paper_attributes.json", "r",encoding='utf-8-sig')
    txt_fp = open("D://data//Paper recommendation system//paper_attributes.txt", 'w', encoding="utf-8-sig")

    output1 = "../word2vec.model"
    output2 = "../vector.model"
    sentences = []

    data_list = json.load(json_fp)
    punctuation_string = string.punctuation + '\n'

    for data in data_list:
        text = data["title"]+' '+data["abstract"] 
        for i in punctuation_string:
            text = text.replace(i, '')
        if text != '':
            txt_fp.write(text + '\n')

    # 6.关闭两个文件
    json_fp.close()
    txt_fp.close()

    with open("D://data//Paper recommendation system//paper_attributes.txt", 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            if " " in line:
                sentences.append(line.split(" "))

    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4, sg=1)  # 定义word2vec 对象

    model.build_vocab(sentences)  # 建立初始训练集的词典
    model.train(sentences, total_examples=model.corpus_count, epochs=10)  # 模型训练

    model.save(output1)  # 模型保存
    model.wv.save_word2vec_format(output2, binary=False)  # 词向量保存


def test():
    model = Word2Vec.load("../word2vec.model")
    word = model.wv.most_similar("ATP")
    for t in word:
        print(t[0], t[1])

if __name__ == "__main__":
    test()
