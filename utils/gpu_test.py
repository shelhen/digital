# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: gpu_test.py
@Project: digital 
@Time: 2025/02/12  06:22
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief: 插入一段描述。
"""
import os
import platform
import tensorflow as tf



def windows_gpu_test():
    print(tf.__version__)  # 查看tensorflow版本
    print(tf.__path__)  # 查看tensorflow安装路径
    a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用
    b = tf.test.is_gpu_available(
        cuda_only=False,
        min_cuda_compute_capability=None
    )  # 判断GPU是否可以用

    print(a)  # 显示True表示CUDA可用
    print(b)  # 显示True表示GPU可用

    # 查看驱动名称
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")


def cpu_test():
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))
    print(tf.__version__)
    # cifar = tf.keras.datasets.cifar100
    # (x_train, y_train), (x_test, y_test) = cifar.load_data()
    # model = tf.keras.applications.ResNet50(
    #     include_top=True,
    #     weights=None,
    #     input_shape=(32, 32, 3),
    #     classes=100, )
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    # model.fit(x_train, y_train, epochs=5, batch_size=64)

    print(tf.config.list_physical_devices())


def platform_test():
    # macOS-15.3.1-arm64-arm-64bit-Mach-O
    print(platform.platform())

def xgboost_test():
    import xgboost as xgb
    print(xgb.__version__)

if __name__ == '__main__':
    # windows_gpu_test()
    # cpu_test()
    # platform_test()
    xgboost_test()
