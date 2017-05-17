# coding=utf8

import numpy as np
import pandas as pd

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)

minimum_price = np.min(data['MEDV'])

a = 900000000000090

print "Minimum price: ${:,.2f}".format(a)
print "Minimum price: ${}".format(minimum_price)