import json
import csv
import string
from gensim.models import Word2Vec

"""
需求：将json中的数据转换成csv文件
"""


def combine():

    json_fp1 = open(
        "D://data//Paper recommendation system//predicate.json", "r", encoding='utf-8-sig')
    json_fp2 = open(
        "D://data//Paper recommendation system//valid_data.json", "r", encoding='utf-8-sig')

    predicate = json.load(json_fp1)
    valid_data = json.load(json_fp2)
    write_list = []
    write_dict = {}
    array=[]
    i=0

    for i in range(9032):

        write_dict = {
            "pub_id": valid_data[i]["id"], "experts": array}
        write_list.append(write_dict)

    with open("D://data//Paper recommendation system//predication_2.json", 'w', encoding='utf-8-sig', errors='ignore') as file:
        json.dump(write_list, file, ensure_ascii=False)


if __name__ == "__main__":
    combine()
