# -*- coding:utf-8 -*-
# @Time       :2021/2/3 23:26
# @Author     :Xing CHEN
# @Site       :
# @File       :drawf1score.py
# @Software   :PyCharm
# @Dirction   :None



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy.interpolate import make_interp_spline



def draw_f1score():
    x = np.array([1,2,3,4,5,6,7])
    precision = np.array([0.91184751214545111,0.80024999999999966,0.70228690476190373,0.63186739094239134,0.56388843917800618,0.52925019057613003,0.46343913072651492])
    precision = np.array([round(i,3) for i in precision])
    recall = np.array([0.17000000000000046,0.4756666666666661,0.6243333333333337,0.6813333333333325,0.7686666666666652,0.74031287605295,0.8046666666666649])
    # [0.2865804066543438, 0.5968652037617556, 0.6607058823529411, 0.6555856816450876, 0.6507366841710428, 0.6169582348305752, 0.5878785488958991]
    recall = np.array([round(i,3) for i in recall])
    # print([2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(x))])
    f1score = np.array([round(2 * precision[i] * recall[i] / (precision[i] + recall[i]),3) for i in range(len(x))])
    # print(f1score)
    # x_smooth = np.linspace(x.min(), x.max(), 7)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    # precision_smooth = make_interp_spline(x, precision)(x_smooth)
    # recall_smooth = make_interp_spline(x, recall)(x_smooth)
    # f1score_smooth = make_interp_spline(x, f1score)(x_smooth)


    fig = plt.figure(figsize = (7,5))  #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
    pl.plot(x,precision, 'g-',marker = '*',label=u'precision score',linestyle = '--',linewidth=1)
    pl.plot(x,recall, 'b-',marker = 'p',label = u'recall score',linestyle='-.',linewidth=1)
    pl.plot(x,f1score, 'r-',marker = 'o',label = u'f1 score',linewidth=1)
    # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    pl.legend()
    pl.xlabel(u'Number of image retrieval')
    pl.ylabel(u'Score of each index')
    # plt.title('AIC-ICC dataset loss')



    for a, b in zip(x, recall):
        plt.text(a+0.3, b, b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(x, f1score):
        plt.text(a+0.3, b, b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(x, precision):
        plt.text(a+0.3, b, b, ha='center', va='bottom', fontsize=10)

    pl.savefig('f1score-1.png')
    pl.show()

if __name__ == '__main__':
    draw_f1score()