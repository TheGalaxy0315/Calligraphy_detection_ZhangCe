# 书法识别  辽宁工程技术大学  软件研23-1张策472321766

<div align="center">
  < img src="https://github.com/TheGalaxy0315/Calligraphy_detection_ZhangCe/blob/master/HOG的训练结果.png">
</div>

47## 项目描述

本项目是一个书法字体风格识别器，通过输入图片，识别出图片中的书法字体风格。项目包含以下文件：
- `0_setting.yaml`：配置文件，包含书法字体风格列表、图片调整大小的目标尺寸等设置。
- `1_Xy.py`：预处理图像、生成训练和测试数据集。
- `2_fit.py`：使用LazyClassifier评估多个分类模型，选择F1分数最高的模型并保存。
- `3_predict.py`：创建一个简单的图形用户界面，用户可以选择图像，程序会显示预测的书法字体风格。
- `util.py`：包含一些辅助功能，例如图像预处理、保存和加载文件等。
改进的特点:对比使用了传统把图像展平，HOG和VGG三种处理方式下的lazypredict训练结果

## 功能

1. 预处理图像并生成训练和测试数据集。
2. 使用了传统把图像展平，HOG和VGG三种方式分别进行预处理
3. 使用LazyClassifier评估多个分类模型，选择F1分数最高的模型并保存。
4. 创建一个简单的图形用户界面，用户可以选择图像，程序会显示预测的书法字体风格。

## 依赖

- Python
- Scikit-learn
- LazyPredict
- OpenCV
- PIL
- Tkinter
- PyYAML
- tensorflow
## 使用

1. 确保已安装所有依赖库。
2. 运行 `1_Xy.py` 生成训练和测试数据集。
3. 运行 `2_fit.py` 评估多个分类模型并保存最佳模型。
4. 运行 `3_predict.py` 启动图形用户界面，选择图像进行预测。


