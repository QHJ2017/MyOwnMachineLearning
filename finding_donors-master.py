# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from time import time

# 导入人口普查数据
data = pd.read_csv("census.csv")

# 成功 - 显示第一条记录
# print data.head()

# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis=1)

# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
cols = ['workclass', 'education_level',
        'marital-status', 'occupation',
        'relationship', 'race',
        'sex', 'native-country']

# TODO：将'income_raw'编码成数字值
features = pd.get_dummies(features_raw, columns=cols)

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
# print "{} total features after one-hot encoding.".format(len(encoded))

le = LabelEncoder()
le.fit(["<=50K", ">50K"])
income = le.transform(income_raw)

X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0)


# X_train, X_test 为特征集，y_train, y_test为标签集


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):  # sample_size,
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set 
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # TODO：使用sample_size大小的训练数据来拟合学习器
    start = time()  # 获得程序开始时间
    learner = learner.fit(X_train, y_train)  # learner学习训练集
    end = time()  # 获得程序结束时间

    # 计算训练时间
    results['train_time'] = end - start

    # TODO: 得到在测试集上的预测值
    #       然后得到对前300个训练数据的预测结果
    start = time()  # 获得程序开始时间
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[0:sample_size])
    end = time()  # 获得程序结束时间

    # TODO：计算预测用时
    results['pred_time'] = end - start

    # TODO：计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train[0:sample_size], predictions_train)

    # TODO：计算在测试集上的准确率
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # TODO：计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train[0:sample_size], predictions_train, 0.5)

    # TODO：计算测试集上的F-score
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)

    # 成功
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    # 返回结果
    return results


clfs = {}

# 高斯朴素贝叶斯 (GaussianNB)
from sklearn.naive_bayes import GaussianNB

clfs["NB"] = GaussianNB()

# 决策树
from sklearn import tree

clfs["DT"] = tree.DecisionTreeClassifier()

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

clfs["AB"] = AdaBoostClassifier()

# 随机森林
from sklearn.ensemble import RandomForestClassifier

clfs["RFC"] = RandomForestClassifier()

# K临近
from sklearn.neighbors import KNeighborsRegressor

# clfs["KN"] = KNeighborsRegressor(n_neighbors=2)
clfs["KN"] = KNeighborsRegressor()

# 随机梯度下降分类器 (SGDC)
from sklearn import linear_model

clfs["SGDC"] = linear_model.SGDClassifier()

# 支撑向量机 (SVM)
from sklearn.svm import SVC

clfs["SVC"] = SVC()

# Logistic回归
from sklearn.linear_model import LogisticRegression

clfs['LR'] = LogisticRegression()

print train_predict(clfs["NB"], 300, X_train, y_train, X_test, y_test)
print train_predict(clfs["DT"], 300, X_train, y_train, X_test, y_test)
print train_predict(clfs["AB"], 300, X_train, y_train, X_test, y_test)
print train_predict(clfs["RFC"], 300, X_train, y_train, X_test, y_test)
# print train_predict(clfs["KN"], 300, X_train, y_train, X_test, y_test)
print train_predict(clfs["SGDC"], 300, X_train, y_train, X_test, y_test)
print train_predict(clfs["SVC"], 300, X_train, y_train, X_test, y_test)
print train_predict(clfs["LR"], 300, X_train, y_train, X_test, y_test)

