# -*- coding:utf-8 -*-
# @Time       :2021/2/3 20:27
# @Author     :Xing CHEN
# @Site       :
# @File       :keywords.py
# @Software   :PyCharm
# @Dirction   :None


import json
import jieba
from jieba import analyse
# 引入TextRank关键词抽取接口
from tqdm import tqdm

test = [
    "一个双臂抬起的运动员跪在绿茵茵的球场上",
    "一个抬着双臂的运动员跪在足球场上",
    "一个双手握拳的男人跪在绿茵茵的足球场上",
    "一个抬起双手的男人跪在碧绿的球场上",
    "一个双手握拳的运动员跪在平坦的运动场上"
  ]
def get_keywords(captions_list):
    keys_set = set()
    captions_str = ''.join(captions_list)
    keywords = analyse.textrank(captions_str,6)
    for key in keywords:
        keys_set.add(key)
    return list(keys_set)


def make_keywords_json(source_path,save_path):


    with open(source_path,encoding='utf-8',mode="r") as f:
        captions_data = json.load(f)

    data = {}
    for i in tqdm(range(len(captions_data))):
        key = captions_data[i]['image_id']
        data[key] = {}
        captions_list = captions_data[i]['caption']
        data[key]['caption'] = captions_list
        data[key]['keywords'] = get_keywords(captions_list)


    with open(save_path, encoding='utf-8', mode ="w") as file:
        json.dump(data,file)

make_keywords_json('caption_train_annotations_20170902.json','train_caption_6keys.json')
# make_keywords_json('caption_validation_annotations_20170910.json','val_caption_5keys.json')