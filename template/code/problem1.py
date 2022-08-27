import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter

train_frame = pd.read_csv('train_titanic.csv')

n = 10
# Bootstrap 采样
train_array = np.array(train_frame)
lenTrain, lenTrainFea = train_array.shape
m = int(lenTrain / 3)
rdmTrain = []
index = [i for i in range(0, lenTrain)]
index = np.array(index)

for i in range(0, n):
    rdmTmp = np.zeros((m, lenTrainFea))
    tmpindex = np.random.choice(index, size=m, replace=True)
    rdmTmp = train_array[tmpindex, :]
    # for j in range(0, m):
    #     tmpindex = np.random.choice(index, replace = True, size = 1)
    #     rdmTmp[j] = train_array[tmpindex, :]
    rdmTrain.append(rdmTmp)
# rdmTrain = np.array(rdmTrain)
# print(rdmTrain[n - 1])


def entropy(label):
    counter = Counter(label)
    ent = 0
    listCo = list(counter)
    length = len(label)
    for i in listCo:
        p = (counter[i] / length)
        ent -= p * math.log2(p)
    return ent


def split(feature, label, dimension):
    # print(feature)
    featureArr = np.array(feature)
    # print(featureArr.shape)
    # print(dimension)
    dimension = int(dimension)
    # dimArr =
    counter = Counter(featureArr[:, dimension])
    listCounter = list(counter)
    count = len(feature)
    split_feature = [[] for i in range(0, len(listCounter))]
    split_label = [[] for i in range(0, len(listCounter))]
    index = 0
    for i in listCounter:
        for j in range(0, count):
            if feature[j][dimension] == i:
                split_feature[index].append(feature[j])
                split_label[index].append(label[j])
        index += 1
    return split_feature, split_label


def best_split(D, A):

    # A‘维数
    D = np.array(D)
    frame = pd.DataFrame(A)
    lenTrain = D.shape[0]
    label = D[:, D.shape[1] - 1]
    k = max(int(math.log2(len(A))), 1)
    feature = np.zeros((lenTrain, 1))
    # print(len(A))
    # print(k)
    tmp = frame.sample(n=k)
    tmp = np.array(tmp)
    # feature = label

    for i in tmp:
        feature = np.c_[feature, D[:, i]]
    featureArr = feature[:, 1:k + 1]
    sizeFeature = k
    length = featureArr.shape[0]
    best_entropy = 0
    best_dimension = -1

    ent = entropy(label)  # 总信息熵

    for i in range(0, sizeFeature):
        # 遍历所有分割
        splitFeature, splitLabel = split(feature, label, dimension=i)
        entNow = 0  # 当前总信息熵
        numSplite = len(splitFeature)
        for j in range(0, numSplite):
            entTemp = entropy(splitLabel[j])
            entNow += len(splitFeature[j]) / length * entTemp

        # 信息增益
        delta = ent - entNow
        if delta > best_entropy:
            best_entropy = delta
            best_dimension = i
        best_dimension = int(tmp[best_dimension])
    return best_dimension


# 记下所有属性可能的取值
D = np.array(train_frame)
A = set(range(D.shape[1] - 1))
possible_value = {}
for every in A:
    possible_value[every] = np.unique(D[:, every])

# 树结点类


class Node:
    def __init__(self, isLeaf=True, label=-1, index=-1):
        self.isLeaf = isLeaf  # isLeaf表示该结点是否是叶结点
        self.label = label  # label表示该叶结点的label（当结点为叶结点时有用）
        self.index = index  # index表示该分支结点的划分属性的序号（当结点为分支结点时有用）
        self.children = {}  # children表示该结点的所有孩子结点，dict类型，方便进行决策树的搜索

    def addNode(self, val: int, node):
        val = int(val)
        self.children[val] = node  # 为当前结点增加一个划分属性的值为val的孩子结点

# 决策树类


class DTree:
    def __init__(self):
        self.tree_root = None  # 决策树的根结点
        self.possible_value = {}  # 用于存储每个属性可能的取值

    '''
    TreeGenerate函数用于递归构建决策树，伪代码参照课件中的“Algorithm 1 决策树学习基本算法”
    '''

    def TreeGenerate(self, D, A: set):

        # 生成结点 node
        node = Node()

        feature = D[:, range(0, D.shape[1] - 1)]
        label = D[:, D.shape[1] - 1]
        counterlab = Counter(label)
        listCo = list(counterlab)

        # if D中样本全属于同一类别C then
        #     将node标记为C类叶结点并返回
        # end if
        if len(listCo) == 1:
            node.isLeaf = True
            node.label = listCo[0]
            return node

        # if A = Ø OR D中样本在A上取值相同 then
        #     将node标记叶结点，其类别标记为D中样本数最多的类并返回
        # end if
        leng = 0
        for i in A:
            leng += len(set(feature[:, i])) - 1
        if feature.shape[1] == 0 or leng == 0:
            node.isLeaf = True
            node.label = counterlab.most_common(1)[0][0]
            return node

        # 从A中选择最优划分属性a_star
        # （选择信息增益最大的属性，用到上面实现的best_split函数）
        a_star = best_split(D, A)

        # for a_star 的每一个值a_star_v do
        #     为node 生成每一个分支；令D_v表示D中在a_star上取值为a_star_v的样本子集
        #     if D_v 为空 then
        #         将分支结点标记为叶结点，其类别标记为D中样本最多的类
        #     else
        #         以TreeGenerate(D_v,A-{a_star}) 为分支结点
        #     end if
        # end for

        node.isLeaf = False
        node.index = a_star
        # print(feature)
        # print(a_star)
        splitFea, splitLab = split(feature, label, a_star)
        allType = self.possible_value[a_star]
        lenSp = len(splitLab)
        for i in allType:
            newNode = Node()
            node.addNode(i, newNode)
            node.children[i].isLeaf = True
            node.children[i].label = counterlab.most_common(1)[0][0]

        for i in range(0, lenSp):
            Dv = np.c_[splitFea[i], splitLab[i]]
            Av = set.copy(A)
            Av.remove(a_star)
            node.children[splitFea[i][0][a_star]] = self.TreeGenerate(Dv, Av)

        return node

    '''
    train函数可以做一些数据预处理（比如Dataframe到numpy矩阵的转换，提取属性集等），并调用TreeGenerate函数来递归地生成决策树
    '''

    def train(self, D):
        D = np.array(D)  # 将Dataframe对象转换为numpy矩阵（也可以不转，自行决定做法）
        A = set(range(D.shape[1] - 1))  # 属性集A

#         #记下每个属性可能的取值
        for every in A:
            self.possible_value[every] = np.unique(train_array[:, every])

        # 递归地生成决策树，并将决策树的根结点赋值给self.tree_root
        self.tree_root = self.TreeGenerate(D, A)

    '''
    predict函数对测试集D进行预测，输出预测标签
    '''

    def predict(self, D, i):
        D = np.array(D)   # 将Dataframe对象转换为numpy矩阵

        #lenTest = D.shape[0]
        #for i in range(0, lenTest):
        x = self.tree_root
        while x.isLeaf == False:
            # 向下查找
            x = x.children[D[i][x.index]]
        res = x.label

        return res
# ----- Your code here -------


dt = [DTree() for i in range(0, n)]
for i in range(0, n):
    dt[i].train(rdmTrain[i])
test_frame = pd.read_csv('test_titanic.csv')

# ----- Your code here -------
right = 0
testArr = np.array(test_frame)
lenTest, lenFea = testArr.shape
lenFea -= 1
for i in range(0, lenTest):
    true = 0
    predict = 0
    for j in range(0, n):
        if dt[j].predict(testArr, i) > 0:
            #print('1')
            true += 1
    if true > lenFea / 2:
        predict = 1
    print(predict, end='\t')
    if predict == testArr[i][lenFea]:
        right += 1
print(right / lenTest)
