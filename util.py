import os
import cv2
import numpy as np
import joblib
import time
import yaml
import skimage
from skimage import feature,exposure
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
# 获取 0_setting.yaml 中的键 key 对应的值 value
def get(key):
    with open("0_setting.yaml", "r",encoding = "utf-8") as f:
        config = yaml.safe_load(f)  # config就自动包含了yaml文件中的所有字典信息
    value = config[key]
    return value

# 预处理图像, 把图像设置为指定大小之后，展平返回
def preprocess_image(file_name, new_size):
    # 1. 读取图像灰度图,并归一化
    img = cv2.imdecode(np.fromfile(os.path.join(file_name),dtype = np.uint8),0)
    img_as_float = skimage.img_as_float(img)
    # 2. 调整图像大小为 new_size
    img_new_size = cv2.resize(img_as_float,new_size)

    # 3. 将图像展平为一维数组
    img=img_new_size.reshape(-1)
    return img
def hog_fliter(file_name,new_size):
    img = cv2.imdecode(np.fromfile(os.path.join(file_name),dtype = np.uint8),0)
    img_as_float=skimage.img_as_float(img)
    img_new_size=cv2.resize(img_as_float,new_size)
    # 计算HOG特征，orientations为方向的数量，pixels_per_cell定义了每个单元格的大小，cells_per_block定义了每个块的大小，visualize=True表示需要返回HOG图像
    # 注意：这里为了更加清晰的展示HOG图像，选择了16*16的小窗大小，正常情况下，为了获取更高的精细程度，可以选择8*8的小窗大小
    hog_vec, hog_edge = feature.hog(img_new_size, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                                    visualize=True)
    # 使用exposure模块的rescale_intensity函数来提高HOG图像的对比度，使得图像更清晰
    hog_edge = exposure.rescale_intensity(hog_edge)
    img=hog_edge.reshape(-1)
    return img
def VGG_fliter(file_name,new_size):
    # 1. 使用keras内置的读图程序，结果为一个PIL图像对象
    img = image.load_img(os.path.join(file_name), target_size=new_size)
    # 2. 将PIL图像对象转换为NumPy数组
    img = image.img_to_array(img)

    # 3. 把单幅图像放到一个数组中，虽然只有一幅图像，但是我们仍然需要扩展数组的维度，以适应VGG16模型的输入尺寸要求（模型要求输入为4D张量）
    X = np.array([img])
    # 有的时候你会看到这样的写法: x = np.expand_dims(x, axis=0) 它与以上的代码是一个意思

    # 4. 使用VGG16模型的预处理函数对图像进行预处理，该步骤包括颜色空间的转换、缩放等
    X = preprocess_input(X)

    # 5. # 加载预训练的VGG16模型，不包括顶部的全连接层（include_top=False），因为我们的目标是提取特征，而不是进行分类
    # weights='imagenet' 表示使用在 ImageNet 数据集上预训练的权重，这些权重可以帮助我们更好地提取特征
    # pooling="max" 表示使用最大池化来池化特征图，这可以帮助我们更好地保留特征信息，并且对结果进行大幅度降维
    model = VGG16(weights='imagenet', include_top=False, pooling="max")

    # 6. 使用 VGG16 模型对图像进行特征提取，model.predict(X) 返回一个包含特征向量的数组
    # [0] 表示我们只提取第一张图像的特征向量，因为我们只输入了一张图像
    y = model.predict(X)[0]
    img = y
    return img


# 用joblib把叫做 name 的对象 obj 保存(序列化)到位置 loc
def dump(obj, name, loc):
    start = time.time()
    print(f"把{name}保存到{loc}")
    joblib.dump(obj,loc)
    end = time.time()
    print(f"保存完毕,文件位置:{loc}, 大小:{os.path.getsize(loc) / 1024 / 1024:.3f}M")
    print(f"运行时间:{end - start:.3f}秒")

# 用joblib读取(反序列化)位置loc的对象obj,对象名为name
def load(name, loc):
    print(f"从{loc}提取文件{name}")
    obj=joblib.load(loc)
    return obj