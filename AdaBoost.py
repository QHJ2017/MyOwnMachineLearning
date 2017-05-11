# -*- coding: utf-8 -*-

'''
这个是我自己写的，可以看到，几乎不是AdaBoost，只是人工模拟了一下。
'''

import math


def Step(TestSet, Labels, E_in, g):
    E = E_in
    e_count = 0
    z = 0.0  # 归一化时使用的Z
    R = []  # R 用来存储划判断对错，正确的存1，错误的存-1
    W = []  # W 用来存储 w
    # 第二步，计算在当前划分中的错误个数
    for i in range(10):
        if Labels[i] != g(TestSet[i]):
            R.append(-1)
            e_count += E[i]
        else:
            R.append(1)
    alpha = 0.5 * math.log((1 - e_count) / e_count)
    # 第三步，计算错分错误率综合，由此计算此弱分类器下的权重alpha
    for i in range(10):
        w = E[i] * math.exp(-1 * alpha * R[i])
        W.append(w)
        z += w
    # print z
    for i in range(10):
        E[i] = W[i] / z
    # print E
    return alpha, E


def AdaBoost(x, TestSet, Labels):
    # x 代表要进行判断的数字，测试集，标签集，layer代表若分类器的个数
    E1 = []  # E 用来存储错误值
    for i in range(10):
        E1.append(0.1)
    alpha_1, E2 = Step(TestSet, Labels, E1, G1)
    alpha_2, E3 = Step(TestSet, Labels, E2, G2)
    alpha_3, E4 = Step(TestSet, Labels, E3, G3)
    # print 'alpha_3: ', alpha_3
    return (alpha_1 * G1(x)) + (alpha_2 * G2(x)) + (alpha_3 * G3(x))


def G1(x):
    if x < 2.5:
        return 1
    else:
        return -1


def G2(x):
    if x < 8.5:
        return 1
    else:
        return -1


def G3(x):
    if x < 5.5:
        return -1
    else:
        return 1


def AdaBoostCalculate(x, layer=3, step=1):
    # x 代表要进行判断的数字，layer代表若分类器的个数
    TestSet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    AdaBoost(1, TestSet, Labels)
    for i in range(10):
        print AdaBoost(i, TestSet, Labels)
    return 0


print AdaBoostCalculate(3)
