# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)  # 扔掉列

minimum_price = np.min(data['RM'])

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print r2_score(y_true, y_pred)


# print data.head()
# print minimum_price