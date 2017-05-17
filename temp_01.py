# coding=utf8

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer  # 其中的 make_scorer 用来创建一个【评分函数】
from sklearn.tree import DecisionTreeRegressor  # 其中的 DecisionTreeRegresso 用来创建一个【决策树的回归函数】
from sklearn.model_selection import GridSearchCV  # 其中的 GridSearchCV 用来创建一个【网格搜索对象】


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    score = r2_score(y_true, y_predict)
    return score


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # 交叉验证的具体方法
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # TODO: Create a decision tree regressor object
    # 决策树的具体方法
    regressor = DecisionTreeRegressor(random_state=0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    # 决策树的10个参数
    params = {'max_depth': range(1, 11)}  # {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    # grid = GridSearchCV(estimator = 决策树, param_grid = 最大深度的10个参数, scoring = R2评价指标, cv = 交叉验证的具体方法)
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# Success
print "Training and testing split was successful."

"""-----------从这里开始是直接复制粘贴的代码。-----------"""
data = pd.read_csv('bj_housing.csv')
prices = data['Value']
features = data.drop('Value', axis=1)
print "BeiJing housing dataset has {} data points with {} variables each.".format(*data.shape)
print data.head()
print "----------"
print prices.head()
print "----------"
print features.head()
print "----------"

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)
print np.shape(X_train)
print np.shape(X_test)

reg = fit_model(X_train, y_train)

print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])

for i, price in enumerate(reg.predict(features)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)