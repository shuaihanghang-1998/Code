import json
import csv
import string
from gensim.models import Word2Vec

"""
需求：将json中的数据转换成csv文件
"""


def combine():

    json_fp0 = open(
        "D://data//Paper recommendation system//peoplevec0.json", "r", encoding='utf-8-sig')
    json_fp1 = open(
        "D://data//Paper recommendation system//peoplevec1.json", "r", encoding='utf-8-sig')
    json_fp2 = open(
        "D://data//Paper recommendation system//peoplevec2.json", "r", encoding='utf-8-sig')
    json_fp3 = open(
        "D://data//Paper recommendation system//peoplevec3.json", "r", encoding='utf-8-sig')
    json_fp4 = open(
        "D://data//Paper recommendation system//peoplevec4.json", "r", encoding='utf-8-sig')

    
    data0 = json.load(json_fp0)
    data1 = json.load(json_fp1)
    data2 = json.load(json_fp2)
    data3 = json.load(json_fp3)
    data4 = json.load(json_fp4)
    write_list = []
    write_dict = {}

    for data in data0:

        write_dict = {
            "id": data["id"], "vec": data["vec"]}
        write_list.append(write_dict)
    for data in data1:

        write_dict = {
            "id": data["id"], "vec": data["vec"]}
        write_list.append(write_dict)
    for data in data2:

        write_dict = {
            "id": data["id"], "vec": data["vec"]}
        write_list.append(write_dict)
    for data in data3:

        write_dict = {
            "id": data["id"], "vec": data["vec"]}
        write_list.append(write_dict)
    for data in data4:

        write_dict = {
            "id": data["id"], "vec": data["vec"]}
        write_list.append(write_dict)

    with open("D://data//Paper recommendation system//peoplevec.json", 'w', encoding='utf-8-sig', errors='ignore') as file:
        json.dump(write_list, file, ensure_ascii=False)


if __name__ == "__main__":
    combine()
