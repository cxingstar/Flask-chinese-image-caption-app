# -*- coding:utf-8 -*-
# @Time       :2021/1/29 11:30
# @Author     :Xing CHEN
# @Site       :
# @File       :make_flickr_json.py
# @Software   :PyCharm
# @Dirction   :None
import json

import numpy as np
def readtxt(caption_file):
    data = []
    id_set = set()
    with open(caption_file,encoding='utf-8',mode = 'r') as f:

        line = f.readline()
        while line:
            items = line.split('#')
            image_id = items[0]
            item = items[2]
            caption = item.split()[1]

            if image_id in id_set:
                data[-1]['caption'].append(caption)
            else:
                filename = {}
                filename['image_id'] = image_id
                filename['caption'] = []
                filename['caption'].append(caption)
                id_set.add(image_id)
                data.append(filename)
            line = f.readline()
    with open('flickr8kval.json',mode='w') as f:
        json.dump(data,f)

