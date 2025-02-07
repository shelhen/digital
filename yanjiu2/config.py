import os
import random
from pathlib import Path
import pandas as pd
import numpy as np



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(12345)



def feature_importance_sensitivity(model, sample):
    """
    通过分别对每个特征施加微小扰动，观察模型预测值的变化程度，考虑了所有模型层的影响，适合非线性模型，但是计算复杂度较高。
    :param model:
    :param sample:输入样本，形状 (n_samples, n_features)
    :return:
    """
    baseline_prediction = model.predict(sample)
    sensitivities = []

    for i in range(sample.shape[1]):
        X_perturbed = sample.copy()
        X_perturbed[:, i] += 1e-4  # 对第 i 个特征增加一个微小扰动
        perturbed_prediction = model.predict(X_perturbed)
        sensitivity = np.mean(np.abs(perturbed_prediction - baseline_prediction))
        sensitivities.append(sensitivity)

    return np.array(sensitivities)


