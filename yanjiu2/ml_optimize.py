# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: ml_optimize.py
@Project: digital_evalutate 
@Time: 2024/12/07  15:08
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief:
树模型的大致调参数思路，先确定大致的参数范围，然后基于网格搜索进行模型精调，若所有参数都放置在较大的参数范围内，则计算时间会非常长。
因此考虑创建几个函数，用于辅助调参，输出更为细致的参数范围。
"""
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, saving
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(123)
random.seed(12345)




tf.random.set_seed(123)


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


def single_param_search(_mod, param_name, _range, **kwargs):
    X_train = kwargs.get("X_train_s")
    X_test = kwargs.get("X_test_s")
    y_train = kwargs.get("y_train")
    y_test = kwargs.get("y_test")
    temp_scores = {}
    for i in _range:
        setattr(_mod, param_name, i)
        _mod.fit(X_train, y_train)
        y_pred = _mod.predict(X_test)
        temp_scores[i] = r2_score(y_test, y_pred)
    # 粗略估计最优值的范围
    _ind = max(temp_scores, key=temp_scores.get)
    return _ind, temp_scores




def total_params_search(param_grid, _mod, **kwargs):
    """
    criterion一般选择基尼系数
    bootstrap，一般选择有放回抽样
    max_features，决策树一般选择'sqrt'，随机森林一般选择p/3，也可以放在模型中搜索。
    n_estimators：分布较广泛，可以处于任意大于1的整数
    max_depth
    min_samples_split
    min_samples_leaf
    # 首先寻找n_estimators:大约在70～100之间，进一步测度约在80-90之间/max_depth处于5～12之间
    """
    X_train_s = kwargs.get("X_train_s")
    X_test_s = kwargs.get("X_test_s")
    y_train = kwargs.get("y_train")
    y_test = kwargs.get("y_test")
    # 十折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=123)
    # 网格搜索
    model = GridSearchCV(_mod, param_grid, cv=kfold, scoring='r2', verbose=0, n_jobs=-1)
    model.fit(X_train_s, y_train)
    print(f"最优模型测试集R2:{r2_score(y_test, model.best_estimator_.predict(X_test_s)):.2%}，测试集MSE：{mean_squared_error(y_test, model.best_estimator_.predict(X_test_s))}")
    return model.best_params_


def model_optimize(**kwargs):
    """希望后台运行此函数，输出全部希望优化的树模型的最优超参数"""
    data_map = {
        "X_train_s": kwargs.get("X_train_s"),
        "X_test_s": kwargs.get("X_test_s"),
        "y_train": kwargs.get("y_train"),
        "y_test": kwargs.get("y_test")
    }

    # 1.决策树模型参数优化
    base_model = DecisionTreeRegressor(random_state=123)
    path = base_model.cost_complexity_pruning_path(data_map["X_train_s"], data_map["y_train"])
    param_grid = {
        'ccp_alpha': path.ccp_alphas,  # 剪枝参数
        'max_depth': np.arange(2, 10),  # 决策树最大深度，用来防止过拟合
        'min_samples_split': np.arange(2, 10),  # 分裂节点所需的最小样本数，也就是如果样本数小于这个值就不划分了。
        'min_samples_leaf': np.arange(1, 5),  # 叶节点所需的最小样本数，如果样本数小于这个，就不划分了。用来防止过拟合
        'max_features': ['sqrt', 'log2']
    }
    model = DecisionTreeRegressor(random_state=123)
    best_params=total_params_search(param_grid, model, **data_map)
    print(f"决策树模型最优超参数:{best_params}")

    # 2.随机森林参数优化：https://www.jianshu.com/p/f5b45a60289f
    param_grid = {
        'max_depth': np.arange(6, 12, 1),  # 树的最大深度
        'n_estimators': np.arange(60, 90, 1),  # 森林中树的数量
        'min_samples_split': [2, 3, 4, 5],  # 分裂节点所需的最小样本数
        'min_samples_leaf': [1, 2, 3],  # 叶节点所需的最小样本数
        'max_features': np.arange(5, 10, 1)  # 每次分裂时考虑的特征数量
    }
    model = RandomForestRegressor(verbose=0, n_jobs=-1, random_state=123)
    best_params = total_params_search(param_grid, model, **data_map)
    print(f"随机森林最优超参数:{best_params}")

    # 3.xgboost参数优化：https://www.cnblogs.com/showmeai/p/16037327.html；https://medium.com/ai%E5%8F%8D%E6%96%97%E5%9F%8E/learning-model-xgbregressor%E5%8F%83%E6%95%B8%E8%AA%BF%E6%95%B4-ca3dcebbe23
    param_grid = {
        'n_estimators': np.arange(50, 65, 1),
        'max_depth': np.arange(8, 12, 1),
        'learning_rate': [0.05, 0.1, 0.2],
        "min_child_weight": [1, 2, 3],
        "gamma": [0, 0.1],
        'subsample': np.arange(0.35, 0.41, 0.01),
        'colsample_bytree': np.arange(0.75, 0.81, 0.01),
    }
    params = {
        "objective": 'reg:squarederror',
        "n_estimators": 41,
        "learning_rate": 0.24,
        "subsample": 0.3,
        "max_depth": 10,
        "min_child_weight": 1,
        "gamma": 0,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 0,
        "scale_pos_weight": 1,
    }
    model = XGBRegressor(**params, n_jobs=-1, random_state=123)
    best_params = total_params_search(param_grid, model, **data_map)
    print(f"xgboost最优超参数:{best_params}")

    # 4.catboost参数优化
    param_grid = {
        'iterations': np.arange(300, 400, 20),
        'depth': [1, 2, 8, 12],
        'learning_rate': [0.01, 0.1, 0.2],
        'l2_leaf_reg': [0, 1],
        'bagging_temperature': [0.0, 0.1, 0.3],
        'border_count': np.arange(32, 128, 10)
    }
    params = {
        "iterations": 300,
        "learning_rate": 0.1,
        "depth": 3,
        "l2_leaf_reg": 1,
        'bagging_temperature': 0,
        "border_count": 64
    }
    model = CatBoostRegressor(**params, random_state=123, verbose=0, train_dir=None)
    best_params = total_params_search(param_grid, model, **data_map)
    print(f"最优超参数:{best_params}")


def load_data(features, label_name='因子得分', test_size=0.2, random_state=132):
    # dataset中的数据是未标准化的数据，但是逆向指标已经正向化。
    dataset = pd.read_csv('./data/dataset.csv')
    # 建立企业id与名称的索引
    index_map = dataset["股票简称"].to_dict()
    y_test: pd.Series = dataset[label_name]
    X: pd.DataFrame = dataset[features].copy(deep=True).astype("float")
    # 数据预处理：1.极差标准化；2.数据集划分。
    X_train, X_test, y_train, y_test = train_test_split(X, y_test, test_size=test_size, random_state=random_state)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    scaler.fit(X)
    # X_train_s = scaler.transform(X)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    return {"X_train_s": X_train_s, "X_test_s": X_test_s, "y_train": y_train, "y_test": y_test}, index_map



if __name__ == '__main__':
    features = [
        '净资产收益率', '投入资本回报率', '成本费用利润率', '总资产周转率', '应收账款周转率', '存货周转率',
        '资产负债率',
        '速动比率', '现金流动负债比率', '营业总收入增长率', '营业利润增长率', '员工收入增长率(%)', '发明专利',
        '研发人员占比(%)', '研发营收比(%)', '股权集中度(%)', '两权分离率(%)'
    ]
    data_map, index_map = load_data(features)
    model_optimize(**data_map)