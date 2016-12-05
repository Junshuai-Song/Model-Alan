# 介绍
样例文件是聚类数目为7的朴素贝叶斯非监督实现，聚类主题数改变直接改变code中K即可。

# 运行环境
* Core: 4 * Intel(R) Core(TM) i5-3317U CPU @ 1.70GHz
* OS: Linux 4.8.4-1-ARCH #1 SMP PREEMPT x86_64 GNU/Linux
* Language: Python 2.7
* Data Set: Activity Recognition from a Single Chest-Mounted Accelerometer

# 问题描述
对输入数据 [X1, X2...Xn ] 进行聚类。注:每个 Xi 表示一个 L 维度的样本,样本个数为 N。使用 聚类结果与原有数据类别做比较,查看聚类效果。

* 输入数据
注:我们将数据集切分,0.8 进行训练,0.2 进行测试 2. 期望输出
(1)对于每个样本聚类的结果; (2)与原有数据类别相比的聚类效果(训练集与测试集)。
X1 =[x11,x12...x1L] X2 =[x21,x22...x2L] ...
XN =[xN1,xN2...xNL]

# 第一部分:朴素贝叶斯非监督EM算法推导

![](https://github.com/songjs1993/model/edit/master/Naive_Bayesian_EM/1.gif)  

![](https://github.com/songjs1993/model/edit/master/Naive_Bayesian_EM/2.gif)  

# 第二部分：测试

1. 数据预处理
• 数据集：
• 取数据集中第一个文件 1.csv 作为处理数据; 
• 类别 0 只有一个样本,将其作为异常点删除;
• 离散化处理各属性;
• 划分数据为 80% 训练集,20% 验证集。

2. 实验过程
我们目标是利用 EM 算法进行样本聚类。
注:因为聚类结果没有顺序,所以对于每一个聚类类别,我们将在此聚类中出现实际类别最多的 类作为其分类;如聚类 1 中出现的实际类的个数为:1 类别 10 个,2 类别 20 个,3 类别 0 个, 4 类别 50 个,那么我们认为此聚类类别 1 代表实际类别 4。

3. 实验结果
我们采用 300 轮迭代充分收敛的聚类精度作为分析。聚类
EM 训练集，聚类7时：73.78%
EM 测试集，聚类7时：73.75%
