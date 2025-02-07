# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: main.py
@Project: common
@Time: 2024/11/19   01:58
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
@Brief: 与评价方法相关的都放在这个模块中。
"""
import pandas as pd
import numpy as np


def _entropy(matrix):
    """
    定义熵权法函数，返回综合得分。
    :param matrix: matrix为需要进行熵权法的pandas
    :return:返回估计出来的权重矩阵
    """
    matrix_ = matrix.copy(deep=True).replace(0, 0.0001)
    for name, item in matrix.items():
        max_ = item.max()
        min_ = item.min()
        for index, row in item.items():
            matrix_.loc[index, name] = (max_ - matrix_.loc[index, name]) / (max_ - min_)
    project, indicator = matrix_.shape
    p = pd.DataFrame(np.zeros([project, indicator], dtype='float64'), columns=matrix_.columns)
    for i in range(indicator):
        p.iloc[:, i] = matrix_.iloc[:, i] / matrix_.iloc[:, i].sum()
    E = -1 / np.log(project) * (p * np.log(p)).sum()
    return (1 - E) / sum((1 - E))


def entropy(matrix):
    _matrix = matrix.copy(deep=True).replace(0, 0.001)
    _matrix = (_matrix - _matrix.min())/ (_matrix.max() - _matrix.min())
    p, i = _matrix.shape
    _matrix = _matrix / _matrix.sum()
    _matrix.replace(0, 0.0001, inplace=True)
    E = -1 / np.log(p) * (_matrix * np.log(_matrix)).sum()
    return (1 - E) / sum((1 - E))


def topsis(matrix, W):
    CCs = []
    m,n =matrix.shape
    pis = np.square(np.ones((m,n)) - matrix)
    nis = np.square(matrix)
    for i in range(m):
        posd = np.sqrt((pis.iloc[i, :]*W).sum())
        negd = np.sqrt((nis.iloc[i, :]*W).sum())
        CCs.append(negd / (posd + negd))
    return pd.Series(CCs, index=matrix.index)


def var_contribution(data):
    """
    该方法用于输出特征值、方差贡献率以及累计方差贡献率
    :param data:未标准化的矩阵
    :return:
    """
    labels = data.columns
    X_scale = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    # 按列求解相关系数矩阵，存入cor中(rowvar=0指定按列求解相关系数矩阵)
    cor = np.corrcoef(X_scale, rowvar=False)
    # 求解相关系数矩阵特征值与特征向量，并按照特征值由大到小排序
    # 注意numpy中求出的特征向量是按列排列而非行，因此注意将矩阵转置
    eigvalue, eigvector = np.linalg.eig(cor)
    eigdf = pd.DataFrame(eigvector.T).join(pd.DataFrame(eigvalue, dtype=float, columns=["Eigvalue"]))
    eigdf.index = labels
    # 将特征向量按特征值由大到小排序
    eigdf = eigdf.sort_values("Eigvalue")[::-1]
    eig_df = eigdf[["Eigvalue"]]
    # 计算每个特征值的方差贡献率，存入varcontribution中
    eig_df["Proportion"] = eig_df.apply(lambda x: x/eig_df["Eigvalue"].sum())
    # 累积方差贡献率
    eig_df["Cumulative"] = eig_df["Proportion"].cumsum()
    # 将特征值、方差贡献率，累积方差贡献率写入DataFrame
    return eig_df


def best_worst_fij(matrix, vec):
    """
    准则函数:确定最优值与最劣值
    a ：指标值矩阵
    b ：表示指标是极大型还是极小型的数组
    """
    f = np.zeros((vec.shape[0], 2)) # 初始化最优值矩阵
    for i in range(vec.shape[0]):
        if vec[i] == 'max':
            f[i, 0] = matrix.max(axis=0)[i] # 最优值
            f[i, 1] = matrix.min(axis=0)[i] # 最劣值
        elif vec[i] == 'min':
            f[i, 0] = matrix.min(axis=0)[i] # 最优值
            f[i, 1] = matrix.max(axis=0)[i] # 最劣值
    return f


def vikor(matrix, vec, w, v=0.5):
    """
    VIKOR
    metrix ：指标值矩阵
    vec ：最优值和最劣值矩阵
    w ：指标权重
    y ：是否绘图
    """

    def SR(matrix, f, w):
        """
        计算效用值S_i 和遗憾值 R_i
        a ：指标值矩阵
        b ：最优值和最劣值矩阵
        c ：指标权重
        """
        s = np.zeros(matrix.shape[0])
        r = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            k = 0
            o = 0
            for j in range(matrix.shape[1]):
                k = k + w[j] * abs((f[j, 0] - matrix.iloc[i, j]) / (f[j, 0] - f[j, 1]))  # 与最优值总的相对距离
                u = w[j] * abs((f[j, 0] - matrix.iloc[i, j]) / (f[j, 0] - f[j, 1]))  # 与给定指标最优值的相对距离
                if u > o:
                    o = u  # 求最大遗憾值
                    r[i] = o
            s[i] = k
        return s, r

    def Q(s, r, v):
        """
        计算折中值 Q_i
        s ：效益值数组
        r ：遗憾值数组
        w ：折衷系数
        """
        q = np.zeros(s.shape[0])
        for i in range(s.shape[0]):
            # q[i] = v *(s[i] - min(s)) / (max(s) - min(s)) +(1 - v)*(r[i] - min(r)) / (max(r) - min(r))
            q[i] = v * (s[i] - min(s)) / (max(s) - min(s)) + (1 - v) * (r[i] - min(r)) / (max(r) - min(r))
        return q

    s, r = SR(matrix, best_worst_fij(matrix, vec), w)
    q = Q(s, r, v)
    return s, r, q


def gra(matrix, w):
    # 熵权-灰色关联
    A1 = matrix.max()
    data = (matrix-A1).abs()
    max_ = data.max().max()
    min_ = data.min().min()
    df_r = (min_*w+data)/(max_*w+data)
    return df_r.mean(axis=1)