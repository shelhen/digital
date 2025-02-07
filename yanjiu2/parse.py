# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: parse.py
@Project: digital_evalutate 
@Time: 2024/12/03  02:22
@Author: xieheng
@Email: xieheng@163.com
@Software: PyCharm
--------------------------------------------------------
@Brief: 插入一段描述。
"""
import pandas as pd
from pathlib import Path






def entropy(matrix):
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


