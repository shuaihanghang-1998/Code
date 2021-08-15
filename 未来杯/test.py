import json
import csv
import string
from gensim.models import Word2Vec

"""
需求：将json中的数据转换成csv文件
"""


def combine():

    json_fp1 = open(
        "D://data//Paper recommendation system//answer.json", "r", encoding='utf-8-sig')
    json_fp2 = open(
        "D://data//Paper recommendation system//predicate_test.json", "r", encoding='utf-8-sig')

    predicate = json.load(json_fp1)
    valid_data = json.load(json_fp2)
    array_predicate=[]
    array_valid_data=[]
    i=0

    for i in range(5):
        array_predicate=predicate[i]["experts"]
        array_valid_data=valid_data[i]["experts"]

        sum = 0
        count = 0
        
        for item in array_predicate:

            sum = sum + 1

            for item2 in array_valid_data:

                if item == item2:
                    count=count + 1

        print(count/sum)




if __name__ == "__main__":
    combine()
