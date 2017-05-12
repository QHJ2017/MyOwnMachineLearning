# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer  # 其中的 make_scorer 用来创建一个【评分函数】
from sklearn.tree import DecisionTreeRegressor  # 其中的 DecisionTreeRegresso 用来创建一个【决策树的回归函数】
from sklearn.model_selection import GridSearchCV  # 其中的 GridSearchCV 用来创建一个【网格搜索对象】

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)  # 扔掉列

minimum_price = np.min(data['RM'])

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# print r2_score(y_true, y_pred)

# print data.head()
# print minimum_price

cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

regressor = DecisionTreeRegressor(random_state=0)

params = {i + 1: 0 for i in range(10)}
scoring_fnc = make_scorer
grid = GridSearchCV

print grid.fit(y_true, y_pred)
print params
