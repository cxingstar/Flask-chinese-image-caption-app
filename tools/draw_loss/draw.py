# -*- coding:utf-8 -*-
# @Time       :2021/1/28 21:36
# @Author     :Xing CHEN
# @Site       :
# @File       :draw.py
# @Software   :PyCharm
# @Dirction   :None


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl



def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df['Step'].tolist(),df['Loss'].tolist()

def draw_aic():
    x,y = read_csv('aic-icc-loss.csv')
    fig = plt.figure(figsize = (7,5))  #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
    ax2 = fig.add_subplot(1,2,1)
    # pl.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')
    # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    p2 = pl.plot(x, y,'r-', label = u'AIC-ICC dataset loss')
    pl.legend()
    #显示图例
    # p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
    pl.legend()
    pl.xlabel(u'iteration times')
    pl.ylabel(u'loss values')
    # plt.title('AIC-ICC dataset loss')
    pl.savefig('aic-icc_train_results_loss.png',dpi = 300)
    pl.show()

def draw_flick8k_cn():
    x,y = read_csv('Flickr8kCN-loss.csv')
    fig = plt.figure(figsize = (7,5))  #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
    pl.plot(x,y,'g-',label=u'Flickr8kCN dataset loss')
    # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    pl.legend()
    pl.xlabel(u'iteration times')
    pl.ylabel(u'loss values')
    # plt.title('AIC-ICC dataset loss')
    pl.savefig('Flickr8kCN_train_results_loss.png',dpi = 300)
    pl.show()


if __name__ == '__main__':
    draw_flick8k_cn()