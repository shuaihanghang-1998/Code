import json
import csv
import numpy as np
import string
from gensim.models import Word2Vec


def peopletovec():

    json_people = open(
        "D://data//Paper recommendation system//expert_attributes4.json", "r", encoding='utf-8-sig')

    # 计算valid数据集中论文与train数据集中论文相似度并计算得分
    people_list = json.load(json_people)

    model = Word2Vec.load("../word2vec.model")

    punctuation_string = string.punctuation + '\n'
    write_list = []
    write_dict = {}
    array = []
    count = 0
    sum=1
    vector = np.zeros(100)

    for data in people_list:
        vector = np.zeros(100)
        sum=1
        count += 1
        if count % 1000 == 0:
            print(count)
        if "interests" in data.keys() and data["interests"] != None:
            for keyword in data["interests"]:
                t = keyword["t"]
                w = keyword["w"]
                array = []
                for i in punctuation_string:
                    t = t.replace(i, '')
                array = array+t.split(" ")
                for word in array:
                    if word in model.wv:
                        vector += model.wv[word]*w
                        sum+=w
            vector=vector/sum
            write_dict = {"id": data["id"], "vec": vector.tolist()}
            write_list.append(write_dict)
            continue

        if "tags" in data.keys()and data["tags"] != None:
            for keyword in data["tags"]:
                t = keyword["t"]
                if "w" not in keyword.keys():
                    w=1
                else:
                    
                    w = keyword["w"]
                array = []
                for i in punctuation_string:
                    t = t.replace(i, '')
                array = array+t.split(" ")
                for word in array:
                    if word in model.wv:
                        vector += model.wv[word]*w
                        sum+=w
            vector=vector/sum
            write_dict = {"id": data["id"], "vec": vector.tolist()}
            write_list.append(write_dict)
            continue

    with open("D://data//Paper recommendation system//peoplevec4.json", 'w', encoding='utf-8-sig', errors='ignore') as file:
        json.dump(write_list, file, ensure_ascii=False)


if __name__ == "__main__":

    peopletovec()
    print("finish!")
