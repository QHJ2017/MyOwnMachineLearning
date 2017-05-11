# -*- coding: utf-8 -*-

import numpy as np

# data_mat = np.mat([[1, 2, 0, 0, 0], [6, 7, 8, 1, 10]])
# a = data_mat[:, 0]
# b = data_mat[:, 3]
# print(a)  # [[1] [6]]
# print(a.A)  # [[1] [6]] a.A和a长得一样，有什么差别呢？打印类型看看
# print(type(a))  # <class 'numpy.matrixlib.defmatrix.matrix'>
# print(type(a.A))  # <type 'numpy.ndarray'> 查看API，是这样写的：return 'self' as an'ndarray' object
# print(type(a.A1))  # <type 'numpy.ndarray'> 查看API，是这样写的：return 'self' as a flattened 'ndarray'
# print(a.A > 0)  # [[ True][ True]]
# print(b.A > 0)  # [[ False][ True]]
# print(np.logical_and(a.A > 0, b.A > 0))  # [[False][ True]] 每个值做逻辑与运算
# print(np.nonzero(np.logical_and(a.A > 0, b.A > 0)))
# # (array([1]), array([0])) np.nonzero，return the indices of the elements that are non-zero.第0个元素是第0轴的下标，第1个元素是第1轴的下标
# print(np.nonzero(np.logical_and(a.A > 0, b.A > 0))[0])  # [1] 因为是二维单列向量，取第0轴下标就行
# print '--------------------'
# print np.nonzero([True, True, False, True])
#
# matM = np.mat([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]])
# print matM[(1, 2, 3), 2]

a = []
for i in range(10):
    print i
    a.append(i)

print '----------------------------------'
print 'a[0]:', a[0]

a[3] = 4
print a
