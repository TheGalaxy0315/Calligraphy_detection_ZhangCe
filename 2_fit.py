# 0. 导入必要的库
import time
from lazypredict.Supervised import LazyClassifier
from util import  load,get,dump

# 1. 加载训练集和测试集
X_train, X_test, y_train, y_test = load("X_train, X_test, y_train, y_test", f'{get("Xy_root")}/Xy')
# 2. 使用LazyClassifier进行快速模型评估
print("开始评估所有的模型:")
clf = LazyClassifier()
scores, _ = clf.fit(X_train, X_test, y_train, y_test)
print(scores) # 打印不同模型的评估结果对比

# 3. 获取F1分数最高的模型
# 获取F1分数最高的模型名称
best_model_name = scores['F1 Score'].idxmax()  # 获取F1分数最高行的索引值，即：模型名称
print("F1分数最高的模型是: ", best_model_name)
# 根据模型名称，从模型字典中获取模型对象
best_model = clf.models[best_model_name]  # 根据模型名称，从模型字典中获取模型对象

# 4. 序列化最佳模型
dump(best_model, "最好的F1分数的模型", f'{get("model_root")}/best_model')