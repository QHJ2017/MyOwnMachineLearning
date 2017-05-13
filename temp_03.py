# -*- coding: utf-8 -*-

# Import libraries necessary for this project
# 载入此项目所需要的库
import numpy as np
import pandas as pd

# Pretty display for notebooks
# 让结果在notebook中显示

# Load the Boston housing dataset
# 载入波士顿房屋的数据集
from sklearn.cross_validation import ShuffleSplit, train_test_split

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# Success
# 完成
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
print data.head()
# RM: 住宅平均房间数量
# LSTAT: 区域中被认为是低收入阶层的比率
# PTRATIO: 镇上学生与教师数量比例
# MEDV: 35年来市场的通货膨胀效应
print type(data)
print data['MEDV'].head()

# TODO: Import 'r2_score'
from sklearn.metrics import r2_score


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)

    # Return the score
    return score


# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)

# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

from sklearn.metrics import make_scorer  # 其中的 make_scorer 用来创建一个【评分函数】
from sklearn.tree import DecisionTreeRegressor  # 其中的 DecisionTreeRegresso 用来创建一个【决策树的回归函数】
from sklearn.model_selection import GridSearchCV  # 其中的 GridSearchCV 用来创建一个【网格搜索对象】


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    #
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1, 11)}  # {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])
