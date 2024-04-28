import tkinter as tk  # Python的标准图形用户界面库，用于创建图形用户界面。
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageTk  # Python图像处理库，提供图像处理功能。

from util import VGG_fliter, load, get #（在使用传统的把图像展平的方法时，VGG_fliter替换为preprocess_image;在使用HOG时，VGG_fliter替换为hog_fliter）
import yaml
char_styles = get('char_styles')  # 字体样式
new_size = get('new_size')  # 新尺寸


class ImageClassifierApp:
    def __init__(self, model_path):
        # 使用util.load加载最佳模型
        self.model =load("最好的F1分数的模型", f'{get("model_root")}/best_model')
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title('Image Classifier')
        self.root.geometry("300x200")

        # 创建一个按钮用于选择图像
        self.button = tk.Button(self.root, text='选择图像', command=self.select_image)
        self.button.pack()

        # 创建一个标签用于显示图像
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # 创建一个标签用于显示预测的类别
        self.prediction_label = tk.Label(self.root)
        self.prediction_label.pack()

        # 启动GUI事件循环
        self.root.mainloop()

    def select_image(self):
        # 打开文件对话框以选择图像
        image_path = filedialog.askopenfilename()

        # 使用util的preprocess_image函数预处理图像
        img_test=np.array(VGG_fliter(image_path,new_size)).reshape(1,-1)#（在使用传统的把图像展平的方法时，VGG_fliter替换为preprocess_image;在使用HOG时，VGG_fliter替换为hog_fliter）

        # 使用加载的最佳模型执行推理
        predicted_class = self.model.predict(img_test)

        # 用PIL读取原图，并设置读取图像后的窗口的大小
        pil_image=Image.open(image_path)
        new_size_window=(100,100) # 设置窗口大小（宽度 x 高度）
        image_resized = pil_image.resize(new_size_window)
        image_resized.show()

        # 将PIL图像转换为PhotoImage并更新标签
        image_tk = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

        # 更新预测标签
        self.prediction_label.config(text=f'预测类别: {char_styles[predicted_class[0]]}')


# 启动应用程序
app = ImageClassifierApp(f'{get("model_root")}/best_model')