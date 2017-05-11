# coding=utf-8
import numpy as np
import pandas as pd
import operator
from IPython.display import display

# 数据可视化代码
from titanic_visualizations import survival_stats
from IPython.display import display

# 加载数据集
in_file = 'titanic_data_qiu_test.csv'
full_data = pd.read_csv(in_file)  # <class 'pandas.core.frame.DataFrame'>
in_file_test = 'titanic_data_qiu.csv'
full_data_test = pd.read_csv(in_file_test)

# 显示数据列表中的前几项乘客数据
# display(full_data_test.head())
print


def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """

    # Ensure that the number of predictions matches number of outcomes
    # 确保预测的数量与结果的数量一致
    if len(truth) == len(pred):

        # Calculate and return the accuracy as a percent
        # 计算预测准确率（百分比）
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)

    else:
        return "Number of predictions does not match number of outcomes!"


'''
处理数据（拆分特征列和待预测列）
'''

# 从数据集中移除 'Survived' 这个特征，并将它存储在一个新的变量中。
outcomes = full_data['Survived']
# print 'outcomes:', type(outcomes)
# print outcomes
data = full_data.drop('Survived', axis=1)
outcomes_test = full_data_test['Survived']
test_data = full_data_test.drop('Survived', axis=1)
# print 'test_data:', test_data
# print '----------------------------'
# 显示已移除 'Survived' 特征的数据集
# print 'data: '
# display(data.head())
# print outcomes

'''
预测
'''


def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    predictions = []
    for _, passenger in data.iterrows():
        if passenger[3] == 'female' or (passenger[3] == 'male' and passenger[4] <= 10):
            predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)


# survival_stats(data, outcomes, 'Sex')
predictions = predictions_2(data)
print '妇女儿童存活几率：', accuracy_score(outcomes, predictions), '\n'


def predictions_2Tree(data):
    predictions = []
    for _, passenger in data.iterrows():
        if passenger[3] == 'female':
            if passenger[1] == 1:  # 阶层1
                if passenger[3] >= 10:
                    predictions.append(1)
                else:
                    predictions.append(0)
            else:
                predictions.append(0)
        else:  # male
            predictions.append(0)
    return pd.Series(predictions)


predictions = predictions_2Tree(data)
print '我的存活率：', accuracy_score(outcomes, predictions), '\n'


def predictions_point(inX, dataSet, labels):  # inX 是一行数据
    matrixData = np.mat(dataSet)
    # print 'matrixData: ', matrixData.shape
    # print matrixData
    dataSetSize = dataSet.shape[0]  # 显示dataSet的函数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # print 'diffMat: ', diffMat.shape
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    distanceD = []
    for i in range(dataSet.shape[0]):
        distanceD.append(distance[i])
    distanceD_2 = np.array(distanceD)
    distanceSorted = distanceD_2.argsort()
    classCount = {}
    for i in range(6):  # K值 6 -> 67%
        voteIlabel = labels[distanceSorted[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def predictions_qiu(testSet, dataSet, outcomes):
    testSet_outcomes = []
    for i in range(testSet.shape[0]):
        testSet_outcomes.append(predictions_point(testSet[i], dataSet, outcomes))
    outcomes_final = pd.Series(testSet_outcomes)
    # print 'outcomes_final:', outcomes_final
    return outcomes_final

# predictions = predictions_point([291, 1, 0, 26, 0, 0, 78.85, 3], data, outcomes)

# print accuracy_score(outcomes_test, predictions_qiu(np.array(test_data), data, outcomes))
# Predictions have an accuracy of 58.92%.
# print 'predictions_qiu(np.array(data), data, outcomes):', predictions_qiu(np.array(data), data, outcomes)
# print '-----------------------------------------'
# print 'outcomes_test:', outcomes_test
