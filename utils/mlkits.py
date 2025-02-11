# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: mlkits.py
@Project: digital 
@Time: 2025/02/08  11:43
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief: 插入一段描述。
"""
import tensorflow as tf
from tensorflow.keras import backend, saving


@saving.register_keras_serializable(package="Custom")
def r2_score(y_true, y_pred):
    """
    r2还可以用mse来计算.
    y_mean = np.mean(y_true)
    r2 = 1 - (mse * len(y_true)) / np.sum((y_true - y_mean)**2)
    :param y_true:
    :param y_pred:
    :param sample_weight:
    :return:
    """
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    # 加入 epsilon 防止除零错误
    return 1 - ss_res / (ss_tot + backend.epsilon())

