# Cudn配置
https://blog.csdn.net/qq_17275369/article/details/140051711

## 1.[GPU 支持指南](https://www.tensorflow.org/install/gpu?hl=zh-cn)
### （1）硬件要求
支持以下带有 GPU 的设备：
- CUDA® 架构为 3.5、5.0、6.0、7.0、7.5、8.0 或更高的 NVIDIA® GPU 卡。请参阅支持 CUDA® 的 GPU 卡列表。
- 如果 GPU 采用的 CUDA® 架构不受支持，或为了避免从 PTX 进行 JIT 编译，亦或是为了使用不同版本的 NVIDIA® 库，请参阅在 Linux 下从源代码编译指南。
- 软件包不包含 PTX 代码，但最新支持的 CUDA® 架构除外；因此，如果设置了 CUDA_FORCE_PTX_JIT=1，TensorFlow 将无法在旧款 GPU 上加载。（有关详细信息，请参阅应用兼容性。）

### （2）软件要求
必须在系统中安装以下 NVIDIA® 软件：
- NVIDIA® GPU 驱动程序 - CUDA® 11.2 要求 450.80.02 或更高版本。
- CUDA® 工具包：TensorFlow 支持 CUDA® 11.2（TensorFlow 2.5.0 及更高版本）及CUDA® 工具包附带的 CUPTI。
- cuDNN SDK 8.1.0 cuDNN 版本。
- （可选）TensorRT 6.0，可缩短用某些模型进行推断的延迟时间并提高吞吐量。
### （3）系统配置
- Windows 设置

确保安装的 NVIDIA 软件包与上面列出的版本一致。特别是，如果没有 cuDNN64_8.dll 文件，TensorFlow 将无法加载。如需使用其他版本，请参阅在 Windows 下从源代码构建指南。并将CUDA®、CUPTI 和 cuDNN 安装目录添加到 %PATH% 环境变量中。

```shell
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin;%PATH%
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64;%PATH%
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include;%PATH%
C:\tools\cuda\bin;%PATH%
```
- [Ubuntu配置](https://www.tensorflow.org/install/gpu?hl=zh-cn#ubuntu_1804_cuda_110)

## 2.环境依赖
TensorFlow 2 软件包现已推出支持适用于 Ubuntu 和 Windows的最新稳定版。软件包和python版本及系统深度绑定，其中支持pip安装的软件包包括Python 3.6–3.9之间，本次选择最新的python3.9测试，如下提供适用于python3.9的且支持GPU的软件包的安装地址：

| 系统     | 支持GPU | 地址 |
|--------|-------|----|
| windows | 是     | https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-2.6.0-cp39-cp39-win_amd64.whl |
| windows | 否     | https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow_cpu-2.6.0-cp39-cp39-win_amd64.whl|
| mac os | 否     | https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-2.6.0-cp39-cp39-macosx_10_11_x86_64.whl|
| Linux | 是     | https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.6.0-cp39-cp39-manylinux2010_x86_64.whl|
| Linux | 是     | https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.6.0-cp39-cp39-manylinux2010_x86_64.whl|

```shell
pip install --upgrade pip
pip install tensorflow
# 如果系统返回了张量，则意味着您已成功安装 TensorFlow。
# python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
