import json
import csv
import string
from gensim.models import Word2Vec

"""
需求：将json中的数据转换成csv文件
"""


def txt_json():

    json_fp = open(
        "D://data//Paper recommendation system//paper_attributes.json", "r", encoding='utf-8-sig')
    txt_fp = open("D://data//Paper recommendation system//paper_attributes1.txt",
                  'w', encoding="utf-8-sig")
    txt_fp_valid = open(
        "D://data//Paper recommendation system//paper_attributes2.txt", 'w', encoding="utf-8-sig")
    json_valid = open(
        "D://data//Paper recommendation system//valid_data.json", "r", encoding='utf-8-sig')

    output1 = "../word2vec.model"
    output2 = "../vector.model"
    sentences = []
    sentences_valid = []

    data_list = json.load(json_fp)
    valid_list = json.load(json_valid)
    punctuation_string = string.punctuation + '\n'

    for data in data_list:
        text = data["title"]+' '+data["abstract"]
        for keyword in data["keywords"]:
            text = text+' '+keyword
        for i in punctuation_string:
            text = text.replace(i, '')
        if text != '':
            txt_fp.write(text + '\n')

    for data in valid_list:
        text = data["title"]+' '+data["abstract"]
        for keyword in data["keywords"]:
            text = text + ' ' + keyword
        for i in punctuation_string:
            text = text.replace(i, '')
        if text != '':
            txt_fp_valid.write(text + '\n')

    # 6.关闭文件
    json_fp.close()
    txt_fp.close()
    txt_fp_valid.close()
    json_valid.close()
    with open("D://data//Paper recommendation system//paper_attributes1.txt", 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line=line.strip('\n')
            if " " in line:
                sentences.append(line.split(" "))
    with open("D://data//Paper recommendation system//paper_attributes2.txt", 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line=line.strip('\n')
            if " " in line:
                sentences_valid.append(line.split(" "))
    model = Word2Vec(vector_size=100, window=5, min_count=1,
                     workers=4, sg=1)  # 定义word2vec 对象
    model = Word2Vec.load("../word2vec.model")
    model.build_vocab(sentences + sentences_valid)
    # model.build_vocab(sentences_valid, update=True)  # 建立初始训练集的词典

    model.train(sentences + sentences_valid, total_examples=model.corpus_count,
                epochs=60)  # 模型训练

    model.save(output1)  # 模型保存
    model.wv.save_word2vec_format(output2, binary=False)  # 词向量保存


def test():
    model = Word2Vec.load("../word2vec.model")
    word = model.wv.most_similar("man")
    for t in word:
        print(t[0], t[1])
    word = model.wv.most_similar("computer")

    for t in word:
        print(t[0], t[1])
    print(model.wv.similarity("computer", "computers"))
    print(model.wv.similarity("woman", "man"))
    print(model.wv["TerpenoidsSteroids"])
    


if __name__ == "__main__":
    #txt_json()
    test()
