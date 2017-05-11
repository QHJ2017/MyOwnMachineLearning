# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as LA

'''
此程序是用来解决压缩感知最后一题的代码。
给一个向量范数与矩阵范数的学习链接：http://blog.csdn.net/bitcarmanlee/article/details/51945271
给一个矩阵运算的连接：http://blog.csdn.net/taxueguilai1992/article/details/46581861
'''

theta = np.mat([
    [0.5377, -1.3077, -1.3499, -0.2050, 0.6715, 1.0347, 0.8884],
    [1.8339, -0.4336, 3.3049, -0.1241, -1.2075, 0.7269, -1.1471],
    [-2.2588, 0.3426, 0.7254, 1.4897, 0.7172, -0.3034, -1.0689],
    [0.8622, 3.5784, -0.0631, 1.4049, 1.6302, 0.2939, -0.8095],
    [0.3188, 2.7694, 0.7147, 1.4172, 0.4889, -0.7873, -2.9443]
])

y = np.mat([7.6030, -4.1781, 3.1123, 1.0586, 7.8053]).transpose()
s = np.mat([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).transpose()

lambda_0 = 0.1
err = 99999999.99

# print theta
# print y
# print s

for j in range(50):
    print '循环第', j, '次。'
    for i in range(7):
        s_temp_final = np.zeros([7, 1]) + s  # s_temp_final保存最终改变的s

        s_temp = np.zeros([7, 1]) + s  # s_temp保存s，方便在原基础上+0.1和-0.1
        s_temp[i] = s_temp[i] + 0.1
        err_temp = LA.norm((y - theta * s_temp), 2) + 0.5 * lambda_0 * LA.norm(s_temp, 1)
        if err_temp < err:
            err = err_temp
            s_temp_final = np.zeros([7, 1]) + s_temp
            print '+0.1:s_temp_final:', s_temp_final, 'err =', err

        s_temp = np.zeros([7, 1]) + s
        s_temp[i] = s_temp[i] - 0.1
        err_temp = LA.norm((y - theta * s_temp), 2) + 0.5 * lambda_0 * LA.norm(s_temp, 1)
        if err_temp < err:
            err = err_temp
            s_temp_final = np.zeros([7, 1]) + s_temp
            print '-0.1:s_temp_final:', s_temp_final, 'err =', err

        s = np.zeros([7, 1]) + s_temp_final

        print '第', i, '遍运算：err =', err, 's = ', s.transpose()

'''
知识点：
当你将一个矩阵，s = np.mat([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).transpose()赋给其他变量时，例如：
a = s
此时你对a操作，便是对s操作。
可见只是一个引用，并不是赋值。
'''
