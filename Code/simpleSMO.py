# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 20:20:43 2018

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt

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

def ShowDataSet(dataMat, labelMat):
    data_plus = []#正样本
    data_minus = []#负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_array = np.array(data_plus)#转换为numpy矩阵
    data_minus_array = np.array(data_minus)#转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_array)[0], np.transpose(data_plus_array)[1], c='green')#正样本散点图
    plt.scatter(np.transpose(data_minus_array)[0], np.transpose(data_minus_array)[1], c='blue')#负样本散点图
    plt.show()
    
ShowDataSet(dataArr, labelArr)

def SMOSimple(datamatrix, labelmatrix, C, tolerant, maxiter):
    DataMatrix = np.mat(datamatrix)
    LabelMatrix = np.mat(labelmatrix).transpose()
    b = 0
    m,n = np.shape(DataMatrix)
    Alphas = np.mat(np.zeros((m,1)))
    Iter = 0
    #最多迭代maxiter次
    while Iter < maxiter:
        AlphaPairsChanged = 0
        for i in range(m):
            #步骤1：计算误差Ei
            fXi = float(np.multiply(Alphas,LabelMatrix).T*\
                        (DataMatrix*DataMatrix[i,:].T)) + b
            Ei = fXi - float(LabelMatrix[i])
            #优化alpha，更设定一定的容错率
            if ((LabelMatrix[i]*Ei < -tolerant) and (Alphas[i] < C)) or \
            ((LabelMatrix[i]*Ei > tolerant) and (Alphas[i] > 0)):
                #随机选择另一个与alpha_i成对优化的alpha_j
                j = SelectJrand(i,m)
                #步骤1：计算误差Ej
                fXj = float(np.multiply(Alphas,LabelMatrix).T*\
                            (DataMatrix*DataMatrix[j,:].T)) + b
                Ej = fXj - float(LabelMatrix[j])
                #保存更新前的aplpha值，使用深拷贝
                AlphasIold = Alphas[i].copy()
                AlphasJold = Alphas[j].copy()
                #步骤2：计算上下界L和H
                if LabelMatrix[i] != LabelMatrix[j]:
                    L = max(0, Alphas[j] - Alphas[i])
                    H = min(C, C + Alphas[j] - Alphas[i])
                else:
                    L = max(0, Alphas[j] + Alphas[i] - C)
                    H = min(C, Alphas[j] + Alphas[i])
                if L == H:
                    print('L==H')
                    continue
                #步骤3：计算eta
                Eta = 2.0 * DataMatrix[i,:]*DataMatrix[j,:].T - \
                DataMatrix[i,:]*DataMatrix[i,:].T - \
                DataMatrix[j,:]*DataMatrix[j,:].T
                if Eta >= 0:
                    print('Eta>=0')
                    continue
                #步骤4：更新alpha_j
                Alphas[j] -= LabelMatrix[j]*(Ei - Ej)/Eta
                #步骤5：修剪alpha_j
                Alphas[j] = ClipAlpha(Alphas[j], H, L)
                if abs(Alphas[j]-AlphasJold) < 0.00001:
                    print('alpha_j变化太小')
                    continue
                #步骤6：更新alpha_i
                Alphas[i] += LabelMatrix[j]*LabelMatrix[i]*\
                (AlphasJold - Alphas[j])
                #步骤7：更新b_1和b_2
                b1 = b - Ei - LabelMatrix[i]*(Alphas[i] - AlphasIold)*\
                DataMatrix[i,:]*DataMatrix[i,:].T - \
                LabelMatrix[j]*(Alphas[j]-AlphasJold)*\
                DataMatrix[i,:]*DataMatrix[j,:].T
                b2 = b - Ej - LabelMatrix[i]*(Alphas[i] - AlphasIold)*\
                DataMatrix[i,:]*DataMatrix[j,:].T - \
                LabelMatrix[j]*(Alphas[j]-AlphasJold)*\
                DataMatrix[j,:]*DataMatrix[j,:].T
                #步骤8：根据b_1和b_2更新b
                if (0 < Alphas[i]) and (C > Alphas[i]):
                    b = b1
                elif (0 < Alphas[j]) and (C > Alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                #统计优化次数
                AlphaPairsChanged += 1
                #打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (Iter,i,AlphaPairsChanged))
        #更新迭代次数
        if AlphaPairsChanged == 0:
            Iter += 1
        else:
            Iter = 0
        print("迭代次数: %d" % Iter)
    return b, Alphas
            
b,alphas = SMOSimple(dataArr, labelArr, 0.6, 0.001, 40)
print(b)
print(alphas)
              
def ShowClassifer(dataMat, labelMat, w, b):
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], c='green')   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], c='blue') #负样本散点图  
    #绘制直线
    x1 = max(dataMat)[0] - 0.35*(max(dataMat)[0]-min(dataMat)[0])
    x2 = min(dataMat)[0] + 0.35*(max(dataMat)[0]-min(dataMat)[0])
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if alpha > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()          
                
def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), \
    np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * \
                dataMat).T, alphas)
    return w.tolist()
                
w = get_w(dataArr, labelArr, alphas)
print(w)
ShowDataSet(dataArr, labelArr)
ShowClassifer(dataArr, labelArr, w, b)
                
                
                
                            
                
            
            
            
            
    
