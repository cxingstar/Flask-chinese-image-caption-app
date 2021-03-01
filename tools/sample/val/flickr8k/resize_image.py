# -*- coding:utf-8 -*-
# @Time       :2021/1/29 12:07
# @Author     :Xing CHEN
# @Site       :
# @File       :resize_image.py
# @Software   :PyCharm
# @Dirction   :None
import os

import cv2

def resize(source_root,save_root,filename):
    img = cv2.imread(source_root+filename,cv2.IMREAD_UNCHANGED)
    dim = (256,256)
    resized = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_root+filename,resized)

if __name__ == '__main__':
    source_root = 'D:/Data/caption/Flickr8k and Flickr8kCN/Flickr8k_image/Flickr8k_Dataset/'
    save_root = 'D:/Data/caption/Flickr8k and Flickr8kCN/Flickr8k_image/Flickr8k_resize/'
    for each in os.listdir(source_root):
        resize(source_root,save_root,each)
# resize(r'667626_18933d713e.jpg')