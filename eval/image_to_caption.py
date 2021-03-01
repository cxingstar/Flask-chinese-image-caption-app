# -*- coding:utf-8 -*-
# @Time       :2021/1/5 22:09
# @Author     :Xing CHEN
# @Site       :
# @File       :image_to_caption.py
# @Software   :PyCharm
# @Dirction   :None
import json


with open('../resource/caption_train_annotations_20170902.json', encoding='utf-8', mode="r") as f:
    captions_data = json.load(f)
print(captions_data[0])

data = {}
for each in captions_data:
    key = each['image_id']
    value = each['caption']
    data[key] = value
with open('../resource/imagecaption_and_keywords.json', encoding='utf-8', mode ="w") as file:
    json.dump(data,file)
