# 介绍
对机器学习、概率图模型、主题模型领域一些模型进行实现，主要涉及一些近年高水平会议论文中提到的算法。

# Model
## 1.EUTB
挖掘stable以及temporal主题：《A Unified Model for Stable and Temporal Topic Detection from Social Media Data》

注：EM-style算法求解

## 2.Naive_Bayesian_EM
朴素贝叶斯的非监督形式实现

## 3.gSpan
频繁子图挖掘算法：《gSpan: Graph-Based Substructure Pattern Mining》

## 4.ullmann
子图同构检测算法：《An Algorithm for Subgraph Isomorphism》

## 5.机器学习_概率解释.pdf
参见：https://github.com/songjs1993/ML/

## 6.UniWalk
Online SimRank：《UniWalk: Unidirectional Random Walk Based Scalable SimRank Computation over Large Graph》
参见：https://github.com/songjs1993/DeepSimRank

## 7. DeepWalk
《DeepWalk: Online Learning of Social Representations》简单说就是将从graph上各顶点Sample出的一些路径看做自然语言处理中的句子，每个顶点看做单词；之后喂到Word2vec中获得每个顶点(单词)的词向量；最后放到一个基本分类器中为graph上顶点做分类。

主要介绍一些常见机器学习算法，同时从损失函数与概率的角度解释了一些机器学习领域内的基本概念。
