# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GMM
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression 
import random
import math
import sys

# reload(sys)   
# sys.setdefaultencoding('utf8')

# mpl.rcParams['font.sans-serif'] = [u'SimHei']
# mpl.rcParams['axes.unicode_minus'] = False

K = 7 

def pre_deal_data(data1,data2):
    # 数据离散化处理
    for i in range(len(data1[0])):
        a = np.array([x[i] for x in data1])	# 取某一列
        sp = np.percentile(a,[x*5 for x in range(21)])
        for j in range(len(data1)):
            #			t = data1[j][i]
            flag = 0
            for k in range(1,len(sp)):
                if(data1[j][i] <= sp[k]):
                    flag = k
                    break
            #			data1[j][i] = k + i*100
            data1[j][i] = flag + i*100
        for j in range(len(data2)):
            #			t = data2[j][i]
            flag = 0
            for k in range(1,len(sp)):
                if(data2[j][i] <= sp[k]):
                    flag = k
                    break
            #			data2[j][i] = k + i*100
            data2[j][i] = flag + i*100


def change(y_hat, y):
    # 聚类结果与真实值结果找对应：例如聚类类别1，可能对应真实类别3
    lable = np.array([[0]*(K+1)]*(K+1))
    #    print(lable.shape)
    for i in range(len(y)):
        lable[int(y_hat[i])][int(y[i])] += 1

    z = [0]*(K+1)
#    print(lable)
#    tot = 0
    for i in range(1,(K+1)):
        y_max = 0
        num_max = -1
        for j in range(1,(K+1)):
            if lable[i][j] > num_max:
                num_max = lable[i][j]
                y_max = j
        z[i] = y_max
    for i in range(len(y_hat)):
        y_hat[i] = z[y_hat[i]]


# 注：数据预处理，将最后一列删掉了


def pre(x_test, phi, pi):
    # 使用训练好的模型对测试集进行预测
    gamma = np.array([[0.0]*K]*len(x_test))
    data = x_test
    for i in range(len(data)):
        tot = 0.0
        for k in range(K):
            #            gamma[i][k] = pi[k] * (phi[0][k][data[i][0]]+lamda) * (phi[1][k][data[i][1]]+lamda) * (phi[2][k][data[i][2]]+lamda)
            gamma[i][k] = pi[k] * (phi[0][k][data[i][0]]) * (phi[1][k][data[i][1]]) * (phi[2][k][data[i][2]])
            tot += gamma[i][k]
        for k in range(K):
            gamma[i][k] /= tot

    y_test_hat = [0.0]*len(data)
    for i in range(len(data)):
        maxc = gamma[i][0]
        lable = 0
        for k in range(1,K):
            if maxc < gamma[i][k]:
                maxc = gamma[i][k]
                lable = k
        y_test_hat[i] = lable + 1
    return np.array(y_test_hat)

def print_ans(gamma, phi, pi, data, y, x_test, y_test, n_iter):
    # 计算预测结果并打印
    y_hat = [0.0]*len(data)
    for i in range(len(data)):
        maxc = gamma[i][0]
        lable = 0
        for k in range(1,K):
            if maxc < gamma[i][k]:
                maxc = gamma[i][k]
                lable = k
        y_hat[i] = lable + 1
    
    y_hat = np.array(y_hat)
    y_test_hat = pre(x_test, phi, pi)

    if n_iter % 1 ==0:
        x_ = np.c_[data, y_hat.reshape(len(y_hat), 1)]
        x_test_ = np.c_[x_test, y_test_hat.reshape(len(y_test_hat), 1)]
        # 朴素贝叶斯分类：多项式
        multinomialNB(x_,y,x_test_,y_test)
        # 朴素贝叶斯分类：高斯分布
        gaussianNB(x_,y,x_test_,y_test)
        logistic(x_,y,x_test_,y_test)
        gmm_deal(x_,y,x_test_,y_test)

    change(y_hat, y)
    change(y_test_hat, y_test)
    acc = np.mean(y_hat.ravel() == y.ravel())
    acc_test = np.mean(y_test_hat.ravel() == y_test.ravel())
    acc_str = u'EM训练集准确率：%.2f%%' % (acc * 100)
    acc_test_str = u'EM测试集准确率：%.2f%%' % (acc_test * 100)
    print(acc_str)
    print(acc_test_str)
    

def gmm_deal(x,y,x_test,y_test):
    gmm = GMM(n_components=7, covariance_type='full', tol=0.0001, n_iter=100, random_state=0)
    gmm.fit(x)

    y_hat = gmm.predict(x)
    y_test_hat = gmm.predict(x_test)
    print(np.min(y_test_hat, axis=0))
    print(np.max(y_test_hat, axis=0))
    
    change(y_hat, y)
    change(y_test_hat, y_test)
              
    # 准确率计算需要改动
    # 对聚类结果，每一类别中实际类别占比最多的，就是当前类别取值
    acc = np.mean(y_hat.ravel() == y.ravel())
    acc_test = np.mean(y_test_hat.ravel() == y_test.ravel())
    print "GMM："
    acc_str = u'训练集准确率：%.2f%%' % (acc * 100)
    acc_test_str = u'测试集准确率：%.2f%%' % (acc_test * 100)
    print acc_str
    print acc_test_str
    
    
def multinomialNB(x,y,x_test,y_test):
    # 多项式分布：朴素贝叶斯
    #create the Multinomial Naive Bayesian Classifier
    clf = MultinomialNB(alpha = 0.01)
    clf.fit(x,y);
    y_hat = clf.predict(x)
    y_test_hat = clf.predict(x_test)
    
    acc = np.mean(y_hat.ravel() == y.ravel())
    acc_test = np.mean(y_test_hat.ravel() == y_test.ravel())
    
    print("朴素贝叶斯：多项式分布")
    acc_str = u'训练集准确率：%.2f%%' % (acc * 100)
    acc_test_str = u'测试集准确率：%.2f%%' % (acc_test * 100)
    print(acc_str)
    print(acc_test_str)

def logistic(x,y,x_test,y_test):
    classifier = LogisticRegression()  # 使用类，参数全是默认的
    classifier.fit(x, y)  # 训练数据来学习，不需要返回值
    y_hat = classifier.predict(x)
    y_test_hat = classifier.predict(x_test)
    acc = np.mean(y_hat.ravel() == y.ravel())
    acc_test = np.mean(y_test_hat.ravel() == y_test.ravel())
    print("logistic回归:")
    acc_str = u'训练集准确率：%.2f%%' % (acc * 100)
    acc_test_str = u'测试集准确率：%.2f%%' % (acc_test * 100)
    print(acc_str)
    print(acc_test_str)

def gaussianNB(x,y,x_test,y_test):
    # 高斯分布朴素贝叶斯
    #create the Multinomial Naive Bayesian Classifier
    clf = GaussianNB()
    clf.fit(x,y);
    y_hat = clf.predict(x)
    y_test_hat = clf.predict(x_test)
    
    acc = np.mean(y_hat.ravel() == y.ravel())
    acc_test = np.mean(y_test_hat.ravel() == y_test.ravel())
    
    print("朴素贝叶斯：高斯分布")
    acc_str = u'训练集准确率：%.2f%%' % (acc * 100)
    acc_test_str = u'测试集准确率：%.2f%%' % (acc_test * 100)
    print(acc_str)
    print(acc_test_str)

def init_with_bayes(pi, phi, data, y):
    # 使用贝叶斯统计信息来初始化EM算法参数: pi / phi
    pi = np.array([0.0] * K)
    for i in range(len(data)):
        pi[int(y[i])-1] += 1
    pi = pi/sum(pi)

    phi = np.array([[[0.0]*21] *K] *3)
    print(phi.shape)
    for i in range(3):
        for j in range(len(data)):
            k = int(data[j][i])
            lable = int(y[j])-1
            phi[i][lable][k] += 1
        for j in range(K):
            phi[i][j] = phi[i][j]/sum(phi[i][j])

def EM():
    data = np.loadtxt('1.csv', dtype=np.float, delimiter=',', skiprows=1)
    #    data = data[50000:70000]
    print(data.shape)
    xx, x, y = np.split(data, [1,4, ], axis=1)
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
    
#   gmm_deal(x,y,x_test,y_test)
    # 朴素贝叶斯分类：多项式
#    multinomialNB(x,y,x_test,y_test)
    # 朴素贝叶斯分类：高斯分布
#    gaussianNB(x,y,x_test,y_test)
#    logistic(x,y,x_test,y_test)

    
    # 预处理，连续数据离散化
    pre_deal_data(x,x_test)
    # 朴素贝叶斯分类：多项式
#    multinomialNB(x,y,x_test,y_test)
    # 朴素贝叶斯分类：高斯分布
#    gaussianNB(x,y,x_test,y_test)
#    logistic(x,y,x_test,y_test)
#    gmm_deal(x,y,x_test,y_test)

    # 朴素贝叶斯非监督分类
    x = x%100
    x_test = x_test%100 #对于前面的k+i*100特征进行处理
    data = x
    n, d = data.shape
    print(data.shape)
    
    # 初始化参数
    # 选择每个类别概率pi
    pi = abs(np.random.standard_normal(K))   # 7个类别[0-6]
    pi = pi/sum(pi)
    # 每个维度属性下：各个类别选择具体取值的多项式分布，随机初始化
    phi = np.array([[[0.0]*21] *K] *3)
    print(phi.shape)
    for i in range(3):
        for j in range(K):
            phi[i][j] = np.array(abs(np.random.standard_normal(21)))
            #            print(phi[i][j])
            phi[i][j] = phi[i][j]/sum(phi[i][j])
    # 每个样本所属各个类别概率
    gamma = np.array([[0.0]*K]*len(data))
    
    # 使用先验统计参数初始化 pi 和 phi
    # init_with_bayes(pi, phi, data, y)
    
    print(pi) # 查看每次是否是初始化
    num_iter = 300
    # EM
    for n_iter in range(num_iter):
        # E
        expect = 0.0
        for i in range(len(data)):
            tot = 0.0
            for k in range(K):
                gamma[i][k] = pi[k] * (phi[0][k][data[i][0]]) * (phi[1][k][data[i][1]]) * (phi[2][k][data[i][2]])
                tot += gamma[i][k]
            for k in range(K):
                gamma[i][k] /= tot
            expect += tot
        
        # M
        # pi
        for k in range(K):
            tot = 0.0
            for i in range(len(data)):
                tot += gamma[i][k]
            pi[k] = tot/len(data)
        
        # phi
        for i in range(3): # 对于第i维数据
            for j in range(len(data)):
                lable = data[j][i]  # 数据j的第i维
                for k in range(K):
                    phi[i][k][lable] += gamma[j][k]
            for k in range(K): # 按行归一化
                tot = 0.0
                for j in range(20):
                    tot += phi[i][k][j]
                for j in range(20):
                    phi[i][k][j] /= tot

        if n_iter % 1 == 0:
            print(n_iter, ":\t", math.log(expect))
            # print(pi,gamma[0])
            print_ans(gamma, phi, pi, data, y, x_test, y_test, n_iter)

    """
    # 朴素贝叶斯分类：多项式
    multinomialNB(x,y,x_test,y_test)
    # 朴素贝叶斯分类：高斯分布
    gaussianNB(x,y,x_test,y_test)
    """


if __name__ == '__main__':
    EM()







