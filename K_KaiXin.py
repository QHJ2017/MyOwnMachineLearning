# -*- coding: utf-8 -*-

from numpy import *
import numpy as np


# 欧氏距离
def eulid_sim(colA, colB):
    return 1.0 / (1.0 + np.linalg.norm(colA - colB))  # np.linalg.norm() 求向量的2范数
    # colC = (colA - colB) ** 2
    # sum = sqrt(colC.sum())
    # return 1.0 / (1.0 + sum)


''' 
# 测试欧氏距离
colA = array([[5, 3, 3]])
colB = array([[1, 1, 1]])
value = eulid_sim(colA, colB)
print "value: ", value
'''


# 皮尔逊相关系数
def pearson_sim(colA, colB):
    if len(colA) < 3:  # 检查是否有3个或更多的点，如果不存在，则返回1，两向量完全相关
        return 1.0
    # return 0.5 + 0.5 * np.corrcoef(colA, colB, rowvar=0)[0][1]
    return np.corrcoef(colA, colB, rowvar=0)
    # np.corrcoef(colA, colB, rowvar=0)[0][1] 返回的是变量的相关系数矩阵，第[0][1]个元素是相关系数，rowvar=0代表列是variables


'''
# 测试皮尔逊相关系数
colA = array([[5, 3, 3, 3, 4, 5], [1, 1, 1, 6, 4, 7]])
colB = array([[1, 1, 1, 6, 7, 7], [5, 4, 3, 3, 4, 5]])
print type(pearson_sim(colA, colB))
'''


# 余弦相似度
def cos_sim(colA, colB):
    num = float(colA.T * colB)
    denom = np.linalg.norm(colA) * np.linalg.norm(colB)
    return 0.5 + 0.5 * (num / denom)


# 计算某个物品和所有其他物品的相似度，进行累加，连评分也累加，最后用累加的总评分／总相似度得到预测该用户对新物品的评分
# data_mat:物品-用户矩阵
# user:用户编号
# item:要预测评分的物品编号
# sim_meas:相似度计算方法
def stand_est(data_mat, user, item, sim_meas):
    n = np.shape(data_mat)[1]  # 取第1轴的元素个数，在这里也就是列数
    sim_total = 0.0
    rat_sim_total = 0.0
    # 遍历整行的每个元素
    for j in range(n):
        user_rating = data_mat[user, j]  # 取一个评分
        if user_rating == 0:  # 如果用户没有评分，就跳过这个物品
            continue
            # 找出要预测评分的物品列和当前取的物品j列里评分都不为0的下标（也就是所有评过这两个物品的用户对这两个物品的评分）
        overlap = np.nonzero(np.logical_and(data_mat[:, item].A > 0, data_mat[:, j].A > 0))[0]
        # 如果预测评分的物品列（假设叫列向量A）和当前取的物品j列（假设叫列向量B）没有都非零的项（也就是说两组向量里要么A评分B没评分，要么B评分A没评分），
        # 则认为相似度为0，否则，计算相似度
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = sim_meas(data_mat[overlap, item], data_mat[overlap, j])
            # 注意overlap是一个array，所以这里还是传的两个列向量，两个元素中都没有0的列向量
        print('the %d and %d similarity is %f' % (item, j, similarity))
        sim_total += similarity  # 累加相似度
        rat_sim_total += similarity * user_rating  # 累加相似度*评分，
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total  # 总评分／总相似度，除以总相似度是为了归一化，将评分归到相似度的范围（比如0～5）


# 用svd将矩阵变换到低维空间，再给出预估评分
# data_mat:物品-用户矩阵
# user:用户编号
# item:物品编号
# sim_meas:相似度计算方法
def svd_est(data_mat, user, item, sim_meas):
    n = np.shape(data_mat)[1]
    sim_total = 0.0
    rat_sim_total = 0.0
    u, sigma, vt = np.linalg.svd(data_mat)  # 对原数据矩阵做svd操作
    sig4 = np.mat(np.eye(4) * sigma[:4])
    # sigma[:4]是取sigma矩阵的前四个，python为了节省空接，sigma矩阵存成了行向量，所以通过eye(4)将其变回对角矩阵
    x_formed_items = data_mat.T * u[:, :4] * sig4.I  # 利用u矩阵将其转换到低维空间,I操作是矩阵的逆.
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0 or j == item:
            continue
        similarity = sim_meas(x_formed_items[item, :].T, x_formed_items[j, :].T)  # 行向量转成列向量
        print('the %d and %d similarity is %f' % (item, j, similarity))
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


# data_mat:数据矩阵
# user:用户编号
# N: 要返回的前N个要推荐的items
# sim_meas: 相似度方法
# est_method:评分预估法
def recommend(data_mat, user, N, sim_meas, est_method):
    un_rated_items = np.nonzero(data_mat[user, :].A == 0)[1]  # 上面说过，nonzero的第1个元素是第1轴的下标，这里也就是列下标
    if len(un_rated_items) == 0:
        return 'you rated everything'
    item_scores = []
    for item in un_rated_items:
        estimate_score = est_method(data_mat, user, item, sim_meas)
        item_scores.append((item, estimate_score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[
           :N]  # 排序，key=lambda jj:jj[1]表示按每个元素的下标为1的参数从大到小排序，取前N个


# data_mat = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# value = stand_est(data_mat, 2, 2, eulid_sim)
# print value


# 用svd将矩阵变换到低维空间，再给出预估评分
# data_mat:物品-用户矩阵
# user:用户编号
# item:物品编号
# sim_meas:相似度计算方法
def svd_est(data_mat, user, item, sim_meas):
    n = np.shape(data_mat)[1]
    sim_total = 0.0
    rat_sim_total = 0.0
    u, sigma, vt = np.linalg.svd(data_mat)  # 对原数据矩阵做svd操作
    print "u", u
    print '--------------------'
    print "sigma:", sigma
    print '--------------------'
    print "vt=", vt
    print '--------------------'
    sig4 = np.mat(np.eye(4) * sigma[:4])  # sigma[:4]是取sigma矩阵的前四个，python为了节省空接，sigma矩阵存成了行向量，所以通过eye(4)将其变回对角矩阵
    x_formed_items = data_mat.T * u[:, :4] * sig4.I  # 利用u矩阵将其转换到低维空间,I操作是矩阵的逆.
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0 or j == item:
            continue
        similarity = sim_meas(x_formed_items[item, :].T, x_formed_items[j, :].T)  # 行向量转成列向量
        print('the %d and %d similarity is %f' % (item, j, similarity))
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


data_mat = mat([[1, 2, 3, 5, 5], [4, 5, 6, 6, 6], [7, 8, 9, 7, 7], [4, 5, 6, 6, 6], [7, 8, 9, 7, 7]])
value = svd_est(data_mat, 1, 1, eulid_sim)
print value



