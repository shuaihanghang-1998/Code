import json
import csv

"""
需求：将json中的数据转换成csv文件
"""
def csv_json():
    # 1.分别 读，创建文件
    json_fp = open("D://data//Paper recommendation system//predicate.json", "r",encoding='utf-8-sig')
    csv_fp = open("D://data//Paper recommendation system//predicate.csv", "w",encoding='utf-8-sig',newline='')
    # 2.提出表头和表的内容
    data_list = json.load(json_fp)
    sheet_title = data_list[0].keys()
    sheet_data = []

    for data in data_list:
        
        #data["abstract"]=data["abstract"][0:30000]
        #避免abstract太长
        sheet_data.append(data.values())

    # 3.csv 写入器
    writer = csv.writer(csv_fp)

    # 4.写入表头
    writer.writerow(sheet_title)

    # 5.写入内容
    writer.writerows(sheet_data)

    # 6.关闭两个文件
    json_fp.close()
    csv_fp.close()


if __name__ == "__main__":
    csv_json()
