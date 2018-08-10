# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 20:20:43 2018

@author: wmy
"""

import numpy as np

'''
函数名：LoadDataSet
输入：文件名
作用：读取数据
返回值：数据列表，标签列表
'''
def LoadDataSet(filename):
    DataMatrix = []
    LabelMatrix = []
    fr = open(filename)
    for line in fr.readlines():
        LineArray = line.strip().split('\t')
        DataMatrix.append([float(LineArray[0]), float(LineArray[1])])
        LabelMatrix.append(float(LineArray[2]))
    return DataMatrix, LabelMatrix 

'''
函数名：SelectJrand
输入：Alpha下标，Alpha总数
'''
def SelectJrand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0,m))
    return j

'''
函数名：ClipAlpha
输入：Alpha
作用：限幅
返回值：Alpha
'''
def ClipAlpha(alpha, H, L):
    #限幅
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L
    return alpha

dataArr, labelArr = LoadDataSet('testSet.txt')
print(dataArr)
print(labelArr)
