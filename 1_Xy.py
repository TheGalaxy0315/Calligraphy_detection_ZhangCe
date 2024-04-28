import glob
import time  # time库，用于时间操作，例如延时
import numpy as np  # numpy库，用于数值计算，简称为np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # tqdm库用于创建进度条
from util import get, VGG_fliter,dump #(在使用传统的把图像展平的方法时，VGG_fliter替换为preprocess_image;在使用HOG时，VGG_fliter替换为hog_fliter)
# 1. 读取配置文件中的信息
train_dir = get("train")  # 获取 训练数据路径
char_styles = get("char_styles")  # 获取 字符样式列表，注意: 必须是列标
new_size = get("new_size")  # 获取 新图像大小元组, 注意: 必须包含h和w
Xy_root = get("Xy_root")
# 2. 生成X,y
print("# 读取训练数据并进行预处理，")
# 使用glob.glob函数查找符合条件的文件，并将结果保存到image_files列表中
X=[]
y=[]
for i in char_styles:
    image_files = glob.glob(f"{train_dir}/train_{i}*")
    for elm in tqdm(image_files,desc = f"处理 {i} 图像：",unit = "it"):
        img = VGG_fliter(elm,new_size)#(在使用传统的把图像展平的方法时：img = preprocess_image(elm,new_size);在使用HOG时:img = hog_fliter(elm,new_size))
        X.append(img)
        label = char_styles.index(i)
        y.append(label)
        time.sleep(0.000001)

X = np.array(X)
y = np.array(y).astype(np.int64)

# 3. 分割测试集和训练集
print("# 将数据按 80% 和 20% 的比例分割")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. 打印样本维度和类型信息
print("X_train: ", X_train.shape, X_train.dtype)  # 训练集特征的维度和类型
print("X_test: ", X_test.shape, X_test.dtype)  # 测试集特征的维度和类型
print("y_train: ", y_train.shape, y_train.dtype)  # 训练集标签的维度和类型
print("y_test: ", y_test.shape, y_test.dtype)  # 测试集标签的维度和类型

# 5. 序列化分割后的训练和测试样本
dump((X_train,X_test,y_train,y_test),"(X_train,X_test,y_train,y_test)","{}/Xy".format(Xy_root))