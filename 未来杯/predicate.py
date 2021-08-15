import json
import csv
import numpy as np
import string
from gensim.models import Word2Vec


def predicate():

    json_paper = open(
        "D://data//Paper recommendation system//paper_attributes.json", "r", encoding='utf-8-sig')
    json_train = open(
        "D://data//Paper recommendation system//train_data_1.json", "r", encoding='utf-8-sig')
    json_valid = open(
        "D://data//Paper recommendation system//valid_data.json", "r", encoding='utf-8-sig')

    # 计算valid数据集中论文与train数据集中论文相似度并计算得分
    paper_list = json.load(json_paper)
    train_list = json.load(json_train)
    valid_list = json.load(json_valid)

    model = Word2Vec.load("../word2vec.model")

    punctuation_string = string.punctuation + '\n'
    valid_text = ''
    paper_text = ''
    valid_array = []
    paper_array = []
    write_list = []
    write_dict = {}
    count = 0

    for valid_data in valid_list:

        valid_keywords_list = valid_data["keywords"]

        max_score = 0
        id = ''
        num = 0

        for paper_data in paper_list:

            score = 0
            paper_keywords_list = paper_data["keywords"]

            for valid_keyword in valid_keywords_list:
                for i in punctuation_string:
                    valid_text = valid_text.replace(i, '')
                valid_array = valid_text.split(" ")
                max = 0
                for paper_keyword in paper_keywords_list:
                    valid_text = valid_keyword
                    paper_text = paper_keyword
                    for i in punctuation_string:
                        paper_text = paper_text.replace(i, '')

                    paper_array = paper_text.split(" ")

                    similarity = model.wv.n_similarity(
                        valid_array, paper_array)

                    if max < similarity:
                        max = similarity

                score = score+max

            valid_text = valid_data["title"]
            paper_text = paper_data["title"]
            for k in punctuation_string:

                valid_text = valid_text.replace(k, '')
                paper_text = paper_text.replace(k, '')

            valid_array = valid_text.split(" ")
            paper_array = paper_text.split(" ")
            score = score+model.wv.n_similarity(valid_array, paper_array)
            #score=score+model.wv.n_similarity(['Forecasting', 'brain', 'storms'], ['Emergence', 'and', 'management', 'of', 'drugresistant', 'enterococcal', 'infections'])

            if max_score < score:
                max_score = score
                id = paper_data["id"]

        for train_data in train_list:
            if id == train_data["pub_id"]:
                # 写入json
                write_dict = {"pub_id": valid_data[id], "experts": train_data["experts"]}
                write_list.append(write_dict)
                count = count+1
                print(count)

    with open("D://data//Paper recommendation system//predicate.json", 'w', encoding='utf-8-sig', errors='ignore') as file:
        json.dump(write_list, file, ensure_ascii=False)


if __name__ == "__main__":

    predicate()
    print("finish!")
