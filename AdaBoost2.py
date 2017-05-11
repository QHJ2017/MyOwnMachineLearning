# -*- coding: utf-8 -*-
import numpy as np


def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                        [1.5, 1.6],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


datMat, classLabels = loadSimpData()


# 通过阈值比较分类，在阈值一边的会被分为-1.0，另一边分为1.0
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 构建了一个数量和dataMatrix第一轴数量一样的值为1的列向量。[[1],[1],[1]……]
    if threshIneq == 'gt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 目的是找特征里最适合做树根的特征
# D：权重向量
def buildStump(dataArr, classLabels, D):  # 训练集，标签，权重矩阵
    dataMatrix = np.mat(dataArr)  # Todo:测试是否能写为 dataMatrix = dataArr
    labelMat = np.mat(classLabels).T  # 转成列向量了
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))  # [[0],[0],[0],...]
    minError = np.inf  # <type 'float'>
    for i in range(n):  # n是第二轴(列)
        rangeMin = dataMatrix[:, i].min()  # 找这一列最小值
        rangeMax = dataMatrix[:, i].max()  # 找这一列最大值
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  # 'lt'是lower than， ‘gt'是greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 阈值设为最小值+第j个步长
                # print('i=%d, threshVal=%f, inequal=%s' % (i, threshVal, inequal))
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 将dataMatrix的第i个特征inequal阈值的置为1，否则为-1
                # print(predictedVals)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0  # 预测对的置0
                # print(errArr)
                weightedError = D.T * errArr
                # print("split: dim %d, threshold %.2f, threshold inequal: %s, the weighted error is %.3f" % (
                #     i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


D = np.mat(np.ones((5, 1)) / 5)
buildStump(datMat, classLabels, D)
