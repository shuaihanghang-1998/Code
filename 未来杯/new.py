import numpy as np
import random
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
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
    vecset=[]
    vecset_valid=[]
    vecset_paper=[]
    valid_array = []
    paper_array = []
    write_list = []
    write_dict = {}
    vector=np.zeros(100)
    flag = 0


    for valid_data in valid_list:
        flag+=1
        if flag % 1000 == 0:
            print(flag)

        sum = 0
        valid_keywords_list = valid_data["keywords"]
        for valid_keyword in valid_keywords_list:
            valid_text=valid_keyword
            for i in punctuation_string:
                valid_text = valid_text.replace(i, '')
            valid_array = valid_array+valid_text.split(" ")
        
        valid_text = valid_data["title"]
        for i in punctuation_string:
            valid_text = valid_text.replace(i, '')
        valid_array = valid_array+valid_text.split(" ")
        for word in valid_array:
            vector += model.wv[word]
            sum += 1
        #vecset.append(vector/sum)
        vecset_valid.append(vector/sum)
        

    for paper_data in paper_list:
        flag+=1
        if flag % 1000 == 0:
            print(flag)

        paper_keywords_list = paper_data["keywords"]
        sum = 0

        for paper_keyword in paper_keywords_list:
            paper_text=paper_keyword
            for i in punctuation_string:
                paper_text = paper_text.replace(i, '')
            paper_array = paper_array+paper_text.split(" ")
        
        paper_text = paper_data["title"]
        for i in punctuation_string:
            paper_text = paper_text.replace(i, '')
        paper_array = paper_array+paper_text.split(" ")

        for word in paper_array:
            vector += model.wv[word]
            sum += 1
        #vecset.append(vector/sum)
        vecset_paper.append(vector/sum)

    print("开始训练")
    hier = AgglomerativeClustering(n_clusters=5000, affinity="euclidean", linkage="average").fit(vecset_valid+vecset_paper) 

    print("训练结束")

    hier_pred_valid = hier.fit_predict(vecset_valid)

    hier_pred_paper = hier.fit_predict(vecset_paper)

    count_valid = 0

    for cla_valid in hier_pred_valid:

        count = 0
        pub_id=valid_data[count_valid]["id"]
        experts=[]

        for cla_paper in hier_pred_paper:

            
            if cla_valid==cla_paper:

                for data in train_list:
                    if data["id"] == paper_list[count]["id"]:
                        experts.append(data["experts"])
           
            count += 1
    
        count_valid += 1
        write_dict = {"pub_id": pub_id, "experts": experts}
        write_list.append(write_dict)


    with open("D://data//Paper recommendation system//predicate_knn.json", 'w', encoding='utf-8-sig', errors='ignore') as file:
        json.dump(write_list, file, ensure_ascii=False)


if __name__ == "__main__":

    predicate()
    print("finish!")
