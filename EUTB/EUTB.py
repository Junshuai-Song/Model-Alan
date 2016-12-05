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


class EUTB():

    iter_tot = 100
    k1 = 10
    k2 = 10
    n = 10   # 调整权重时，滑动窗口大小
    lamda = 0.5 # 空间正则的占比
    yipu = 20 # 时间正则的占比    
    gamma = 0.5 # 牛顿法迭代步长
    alpha = 1.2 # 阈值
    
    user = set()    # user: name -> id 对应
    time = set()    # time: name -> id 对应
    word = set()    # words: name -> id 对应
    user_dict = dict()
    time_dict = dict()
    word_dict = dict()
    # M

    
    def __init__(self):
        with open('./data/checkins.txt') as f:
            self.data = f.readlines()
        self.data = self.data[0:50]

    def pre_deal_w(self):
        """
        预处理统计所有词汇，得到词表；
        同时得到W[u,t,w]词频数据，t按照同一天的计算
        """
        # user & time & word
        for line in self.data:
            line = line.split('\t')
            self.user.add(line[0])         # user
            self.time.add(line[4][:10])    # time
            line = line[6].split('|')      # words
            for w in line:
                self.word.add(w)
        self.user_size = len(self.user)
        self.time_size = len(self.time)
        self.word_size = len(self.word)
        print("user_size = " + str(len(self.user)))
        print("time size = " + str(len(self.time)))
        print("word_size = " + str(len(self.word)))
        
        # 构造user/time/word下标映射
        i = 0
        for line in self.user:
            self.user_dict[line] = i
            i+=1
        i = 0
        for line in self.time:
            self.time_dict[line] = i
            i+=1
        i = 0
        for line in self.word:
            self.word_dict[line] = i
            i+=1

        # 构造M[u,t,w]词频数组
        self.M = np.array([[[0] * len(self.word)] * len(self.time)] * len(self.user))
        print("M.shape = " + str(self.M.shape))
        for line in self.data:
            line = line.split('\t')
            u = self.user_dict[line[0]]         # user
            t = self.time_dict[line[4][:10]]    # time
            line = line[6].split('|')   # words
            for w in line:
                w = self.word_dict[w]
                self.M[u][t][w] += 1
        

    
    def pre_deal_data(self):
        """
        数据预处理：
        （1）分词，统计所有词汇；同时得到W[u,t,w]词频数据，t按照一天算
        （2）获取π(u,v)
        （3）get_burst_degree()，计算改进3需要的预处理内容
        """
        print("pre_deal data...")
        self.pre_deal_w()
        self.get_pai()
        self.get_burst_degree() # 第三项：这个只需要计算一次，放在初始化中
        print("pre_deal data end!\n")

    def get_pai(self):
        # 目前没有找到社交网络上u/v之间好友信息，暂时保留不处理
        # deal with π(u,v)
        
        return 0

    def get_burst_degree(self):
        """
        处理第三个提升，获得burst_degree数组
        """
        # 统计每个词在各个时刻出现的次数，把矩阵M的User积掉即可
        freq = np.array([[0.0]* self.time_size] * self.word_size)
        for i in range(self.word_size):
#            cnt = 0
            for j in range(self.time_size):
                tot = 0
                for k in range(self.user_size):
                    tot += self.M[k][j][i]
                freq[i][j] = tot
#                cnt += tot
#            print(i,cnt)
        """
        for i in range(self.time_size):
            cnt = 0
            for j in range(self.word_size):
                cnt += freq[j][i]
            print(i, cnt)
        print([x[10] for x in freq])
        """
        # 计算burst_degree
        self.burst_degree = np.array([[0.0] * self.time_size] * self.word_size)
        for i in range(self.word_size):
            self.burst_degree[i][0] = self.alpha # 第一个不知道应该赋值什么，先赋值为alpha吧
            for j in range(1, self.time_size):
                # 对窗口n中的数据计算 [i-n, i-1]
                u = 0.0
                for k in range(1, self.n):
                    if j-k<0:
                        break
                    u += freq[i][j-k]   # 这个里面大多数都是0，很有可能前面计算的均值和方差都是0，这样就没办法用
                u /= (float(min(self.n, j+1)))

                thita = 0.0
                for k in range(1, self.n):
                    if j-k<0:
                        break
                    thita += (freq[i][j-k]-u)*(freq[i][j-k]-u)
                thita /= (float(min(self.n, j+1)))
                thita = math.sqrt(thita)
                
                # 取一个max，有超参数alpha的影响
                if thita==0:
                    self.burst_degree[i][j] = self.alpha
                else:
                    self.burst_degree[i][j] = max(abs(freq[i][j] - u)/thita, self.alpha)   
#                    if freq[i][j] > 0:
#                        print(u, thita, self.burst_degree[i][j])
        
        # 打印出burst_degree看一下
#        print(self.burst_degree[1])
        
        return 1

    def iterator_theta(self):
        
        return 0

    
    def spatial_regularization(self):
        """
        第一个正则
        待补充
        """
        # theta_u_new = [[0.0]*self.user_size] * self.k1
        
        return 0
    
    def temporal_regularization(self):
        """
        第二个正则
        处理theta_t二维数组，迭代
        """
        gamma = self.gamma
        theta_t_new =  np.array([[0.0]*self.k2] * self.time_size)
        theta_t_new2 = np.array([[0.0]*self.k2] * self.time_size)
        
        for i in range(len(self.theta_t)):
            for j in range(len(self.theta_t[0])):
                theta_t_new2[i][j] = self.theta_t[i][j]
                theta_t_new[i][j] = self.theta_t[i][j]
        expect = self.com_expect(theta_t_new2)
        expect2 = expect + 0.01
        
        while(expect2 > expect and (expect2-expect)>=0.001):    #期望需要增加，且需要增加足够多
            for i in range(len(theta_t_new2)):
                for j in range(len(theta_t_new2[0])):
                    theta_t_new[i][j] = theta_t_new2[i][j]
            expect = expect2
            for i in range(self.k2):
                theta_t_new2[0][i] = theta_t_new[0][i]
                theta_t_new2[self.time_size-1][i] = theta_t_new[self.time_size-1][i]
            for t in range(1, self.time_size-1):
                for i in range(self.k2):
#                    theta_t_new2[t][i] = (1.0-gamma)*theta_t_new[t][i] + gamma*(theta_t_new[t-1][i] + theta_t_new[t+1][i])/2.0
                    theta_t_new2[t][i] = (1.0-gamma)*theta_t_new[t][i] + gamma*(theta_t_new[t-1][i] + theta_t_new[t+1][i])/2.0

            # 归一化
            for i in range(len(theta_t_new2)):
                tot = 0.0
                for j in range(len(theta_t_new2[0])):
                    tot += (theta_t_new2[i][j])
                for j in range(len(theta_t_new2[0])):
                    theta_t_new2[i][j] /= tot
            expect2 = self.com_expect(theta_t_new2)
            # print("E------:" + str(expect2) + "," + str(expect))
            
        if expect > self.com_expect(self.theta_t):
            for i in range(len(self.theta_t)):
                for j in range(len(self.theta_t[0])):
                    self.theta_t[i][j] = theta_t_new[i][j]
        """
        # 在上面归一化过了
        for i in range(len(self.theta_t)):
            tot = 0.0
            for j in range(len(self.theta_t[0])):
                tot += (self.theta_t[i][j])
            for j in range(len(self.theta_t[0])):
                self.theta_t[i][j] /= tot
        """
        
        return 1
    
    def burst_weighted(self):
        """
        第三个提升
        对phi_t进行权重调整，需要重新归一化
        """
        for w in range(len(self.phi_t[0])):
            t1 = set()
            for t in range(self.time_size):
                if self.burst_degree[w][t]>self.alpha:
                    t1.add(t)
            for k in range(len(self.phi_t)):
                t2 = set()
                maxc = -1.0
                for t in range(self.time_size):
                    if self.theta_t[t][k] > maxc:
                        maxc = self.theta_t[t][k]
                maxc *= 0.9
                for t in range(self.time_size):
                    if self.theta_t[t][k] > maxc:
                        t2.add(t)
                t = t1 & t2 # 取两个set的交集，在这些时间下的主题k，下的词w，
#                print(t)
                maxc = -1.0
                for j in t:
                    if self.burst_degree[w][j] > maxc:
                        maxc = self.burst_degree[w][j]
#                if math.sqrt(max(maxc, self.alpha)) > 1.0:
#                    print(math.sqrt(max(maxc, self.alpha)), self.phi_t[k][w], self.phi_t[k][w]*math.sqrt(max(maxc, self.alpha)))
                self.phi_t[k][w] *= math.sqrt(max(maxc, self.alpha))
#                print(math.sqrt(max(maxc, self.alpha)))
        # 归一化
        for k in range(self.k2):
            tot = 0.0
            for w in range(self.word_size):
                tot += self.phi_t[k][w]
            for w in range(self.word_size):
                self.phi_t[k][w] /= tot

        return 1
    
    def com_expect(self, theta_t_new):
        """
        按照输入的theta_t计算E
        """
        E = 0.0
        for u in range(self.user_size):
            for t in range(self.time_size):
                for w in range(self.word_size):
                    cnt1 = 0.0
                    for k in range(self.k1):
                        cnt1 += self.theta_u[u][k]*self.phi_u[k][w]
                    cnt2 = 0.0
                    for k in range(self.k2):
                        cnt2 += self.theta_t[t][k]*self.phi_t[k][w]
                    E += ((math.log(self.lamda_u * cnt1 + (1-self.lamda_u) * cnt2))*self.M[u][t][w])

#        print("E:" + str(E))
        
        # 下面计算正则
        # 第一项：缺少数据，暂时不加
        tot = 0.0
        
        # 第二项
        tot = 0.0
        for t in range(len(theta_t_new)-1):
            for k in range(self.k2):
                tot += (theta_t_new[t][k] - theta_t_new[t+1][k]) * (theta_t_new[t][k] - theta_t_new[t+1][k])
#        print("err:" + str(tot))                
        E -= self.yipu*tot

        return E
        
    def EM(self):
        """
        开始迭代
        （1）
        （2）
        （3）
        """
        print("lamda = " + str(self.lamda))
        # 随机初始化theta_u/theta_t/phi_u/phi_t
        self.theta_u = np.random.random((self.user_size,self.k1))
        self.theta_t = np.random.random((self.time_size,self.k2))
        self.phi_u = np.random.random((self.k1,self.word_size))
        self.phi_t = np.random.random((self.k2,self.word_size))
        self.lamda_u = np.random.random(1)  # 第一轮随机，之后在M步迭代计算
        print("lamda_u = " + str(self.lamda_u))
        
        iter_num = 0
        while(iter_num < self.iter_tot):
        
            # E
#            print("E...")
            # 对每一个单词来说，计算其属于stable和temporal主题的概率分布；这里数据结构不行，最好是每个u/t下单词的数量，为第三维度的大小
            p_u = np.array([[[[0.0] * self.k1] * self.word_size] * self.time_size] * self.user_size)
            p_t = np.array([[[[0.0] * self.k2] * self.word_size] * self.time_size] * self.user_size)
            """
            B_u = 0.0
            B_t = 0.0
            for line in self.data:
                line = line.split('\t')
                u = self.user_dict[line[0]]         # user
                t = self.time_dict[line[4][:10]]    # time
                line = line[6].split('|')           # words
                for w in line:
                    w = self.word_dict[w]
                    for k in range(self.k1):
                        B_u += self.theta_u[u][k]*self.phi_u[k][w]
                    for k in range(self.k2):
                        B_t += self.theta_t[u][k]*self.phi_t[k][w]
            
            B = self.lamda_u*B_u + (1-self.lamda_u)*B_t
            """
            for line in self.data:
                line = line.split('\t')
                u = self.user_dict[line[0]]         # user
                t = self.time_dict[line[4][:10]]    # time
                line = line[6].split('|')           # words
                words = set()
                for w in line:
                    words.add(self.word_dict[w])
                for w in words:
                    # 已经是换好的序号下标了
                    B_u = 0.0
                    B_t = 0.0
                    for k in range(self.k1):
                        B_u += self.theta_u[u][k]*self.phi_u[k][w]
                    for k in range(self.k2):
                        B_t += self.theta_t[t][k]*self.phi_t[k][w]
                    B = self.lamda_u*B_u + (1-self.lamda_u)*B_t

                    #注意这里对每个单词只计算一次，没用到词频（原数据集中没进行处理，所以这里需要单独处理一下）
                    for k in range(self.k1):    
                        p_u[u][t][w][k] += self.lamda_u * self.theta_u[u][k] * self.phi_u[k][w] / B
                    for k in range(self.k2):
                        p_t[u][t][w][k] += (1-self.lamda_u)*self.theta_t[t][k] * self.phi_t[k][w] / B
                    
            # M
#            print("M...theta_u/t")            
            # (1)正常计算
            self.theta_u = np.array([[0.0] *self.k1] * self.user_size)
            self.theta_t = np.array([[0.0] *self.k2] * self.time_size)
            tot_u = [0.0] * self.user_size
            tot_t = [0.0] * self.time_size
            for line in self.data:
                line = line.split('\t')
                u = self.user_dict[line[0]]         # user
#                print(u, line[0])
                t = self.time_dict[line[4][:10]]    # time
                line = line[6].split('|')           # words
                for w in line:
                    w = self.word_dict[w]
                    for k in range(self.k1):
                        tot_u[u] += p_u[u][t][w][k]
                    for k in range(self.k2):
                        tot_t[t] += p_t[u][t][w][k]

            for line in self.data:
                line = line.split('\t')
                u = self.user_dict[line[0]]         # user
                t = self.time_dict[line[4][:10]]    # time
                line = line[6].split('|')           # words
                for w in line:
                    w = self.word_dict[w]
                    for k in range(self.k1):
                        self.theta_u[u][k] += p_u[u][t][w][k]
                    for k in range(self.k2):
                        self.theta_t[t][k] += p_t[u][t][w][k]
            # 归一化
            for i in range(len(self.theta_u)):
                for k in range(self.k1):
                    self.theta_u[i][k] /= tot_u[i]
            for i in range(len(self.theta_t)):
                for k in range(self.k2):
                    self.theta_t[i][k] /= tot_t[i]

            # phi_u & phi_t
#            print("M...phi_u/t")            
#            self.phi_u = 0.0
#            self.phi_t = 0.0
            self.phi_u = np.array([[0.0] * self.word_size] * self.k1)
            self.phi_t = np.array([[0.0] * self.word_size] * self.k2)
            tot_u = [0.0] * self.k1
            tot_t = [0.0] * self.k2
            for line in self.data:
                line = line.split('\t')
                u = self.user_dict[line[0]]         # user
                t = self.time_dict[line[4][:10]]    # time
                line = line[6].split('|')           # words
                for w in line:
                    w = self.word_dict[w]
                    for k in range(self.k1):
                        tot_u[k] += p_u[u][t][w][k]
                    for k in range(self.k2):
                        tot_t[k] += p_t[u][t][w][k]

            for line in self.data:
                line = line.split('\t')
                u = self.user_dict[line[0]]         # user
                t = self.time_dict[line[4][:10]]    # time
                line = line[6].split('|')           # words
                for w in line:
                    w = self.word_dict[w]
                    for k in range(self.k1):
                        self.phi_u[k][w] += p_u[u][t][w][k]
                    for k in range(self.k2):
                        self.phi_t[k][w] += p_t[u][t][w][k]
            
            # 归一化
            for k in range(len(self.phi_u)):
                for w in range(len(self.phi_u[0])):
                    self.phi_u[k][w] /= tot_u[k]
            for k in range(len(self.phi_t)):
                for w in range(len(self.phi_t[0])):
                    self.phi_t[k][w] /= tot_t[k]
            
            # lamda_u
            tot = 0.0
            self.lamda_u = 0.0
            for line in self.data:
                line = line.split('\t')
                u = self.user_dict[line[0]]         # user
                t = self.time_dict[line[4][:10]]    # time
                line = line[6].split('|')           # words
                for w in line:
                    w = self.word_dict[w]
                    for k in range(self.k1):
                        self.lamda_u += p_u[u][t][w][k]
                        tot += p_u[u][t][w][k]
                    for k in range(self.k2):
                        tot += p_t[u][t][w][k]
            self.lamda_u /= tot

            # (2)两个正则
            #self.spatial_regularization()
#            print("temporal_regularization...")
            self.temporal_regularization()

            # (3)词权重调整
#            print("burst_weighted...")       
#            self.burst_weighted()
            
            if iter_num%1==0:
                # 输出生成概率，保证递增（正确性）
                print("iter_num = " + str(iter_num) + ", E: " + str(self.com_expect(self.theta_t)))

            iter_num += 1
        return 0 

    def print_ans(self):
        """
        对temporal topics输出，输出所有topic的Top K的词来表征这个主题
        """
        ans = self.word_dict.keys()
        top_num = 6
        
        for k in range(self.k2):
            """
            maxc = -1.0
            flag = 0
            for k in range(self.k2):
                if self.theta_t[t][k] > maxc:
                    maxc = self.theta_t[t][k]
                    flag = k
            """
            flag = k
            value = np.array([0.0]*top_num)  #输出前top K的词
            words = np.array([0] * top_num)
            for w in range(self.word_size):
                # 对每一个词，查看其在当前flag主题下，是否在top 20中
                minc = 1000000.0
                wei = 0
                for v in range(len(value)):
                    if(value[v] < minc):
                        minc = value[v]
                        wei = v
                if(self.phi_t[flag][w] > minc):
                    value[wei] = self.phi_t[flag][w]
                    words[wei] = w
            # 输出当前主题下top 20的单词，用来看当前主题如何（默认的时间区间为1天之内，这里查看某天主题分布情况）
            i = 0
            for w in words:
                print (i,ans[w],value[i]),
                i += 1
            print("\n")
        return 1

    def run(self):
        self.pre_deal_data()
        self.EM()
        return 0
    

if __name__ == '__main__':
    eutb = EUTB()
    eutb.run()
    eutb.print_ans()




    


