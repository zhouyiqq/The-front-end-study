# -- encoding:utf-8 --
"""
只要是机器学习中，代码的编写流程一般和下面这个一样！！！！
https://scikit-learn.org/0.18/modules/classes.html#module-sklearn.linear_model
Create on 19/3/2
"""
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc, roc_curve, classification_report

warnings.filterwarnings('ignore')
mpl.rcParams['font.sans-serif'] = [u'simHei']

# 一、数据加载
path = "../datas/iris.data"
names = ['A', 'B', 'C', 'D', 'cla']
df = pd.read_csv(filepath_or_buffer=path, sep=",", header=None, names=names)
# df.info()

# 二、数据的清洗
# class_name_2_label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# df['cla'] = list(map(lambda cla: class_name_2_label[cla], df['cla'].values))
# print(df['cla'].values)
# df.info()

# 三、根据需求和原始模型从最原始的特征属性中获取具体的特征属性矩阵X和目标属性矩阵Y
X = df.drop('cla', axis=1)
X = np.asarray(X).astype(np.float64)
Y = df['cla']
# 对目标属性做一个类别的转换，将字符串的数据转换为从0开始的int值
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
# print(label_encoder.classes_)
# print(label_encoder.transform(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))
# print(label_encoder.inverse_transform([0, 1, 2, 0, 2, 1]))
# X.info()

# 四、数据分割(将数据分割为训练数据和测试数据)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=28)
print("训练数据X的格式:{}, 以及数据类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的数据类型:{}".format(type(y_train)))
print("Y的取值范围:{}".format(np.unique(Y)))

# 五、特征工程的操作
# # a. 创建对象(标准化操作)
# scaler = StandardScaler()
# # b. 模型训练+训练数据转换
# x_train = scaler.fit_transform(x_train, y_train)
# # c. 基于训练好的对象对数据做一个转换
# x_test = scaler.transform(x_test)

# 六、模型对象的构建
"""
def __init__(self,
             criterion="gini",
             splitter="best",
             max_depth=None,
             min_samples_split=2,
             min_samples_leaf=1,
             min_weight_fraction_leaf=0.,
             max_features=None,
             random_state=None,
             max_leaf_nodes=None,
             min_impurity_decrease=0.,
             min_impurity_split=None,
             class_weight=None,
             presort=False)
    criterion: 给定决策树构建过程中的纯度的衡量指标，可选值: gini、entropy， 默认gini
    splitter：给定选择特征属性的方式，best指最优选择，random指随机选择(局部最优)
    max_features：当splitter参数设置为random的有效，是给定随机选择的局部区域有多大。
    max_depth：剪枝参数，用于限制最终的决策树的深度，默认为None，表示不限制
    min_samples_split=2：剪枝参数，给定当数据集中的样本数目大于等于该值的时候，允许对当前数据集进行分裂；如果低于该值，那么不允许继续分裂。
    min_samples_leaf=1, 剪枝参数，要求叶子节点中的样本数目至少为该值。
    class_weight：给定目标属性中各个类别的权重系数。
"""
algo = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_leaf=1, min_samples_split=2)

# 七. 模型的训练
algo.fit(x_train, y_train)

# 八、模型效果的评估
print("各个特征属性的重要性权重系数(值越大，对应的特征属性就越重要):{}".format(algo.feature_importances_))
print("训练数据上的分类报告:")
print(classification_report(y_train, algo.predict(x_train)))
print("测试数据上的分类报告:")
print(classification_report(y_test, algo.predict(x_test)))
print("训练数据上的准确率:{}".format(algo.score(x_train, y_train)))
print("测试数据上的准确率:{}".format(algo.score(x_test, y_test)))

# 查看相关属性
test1 = [x_test[10]]
print("预测函数:")
print(algo.predict(test1))
print("预测概率函数:")
print(algo.predict_proba(test1))
print(algo.apply(test1))

# 决策树可视化
# 方式一：输出dot文件，然后使用dot的命令进行转换
# 命令：dot -Tpdf iris.dot -o iris.pdf 或者dot -Tpng iris.dot -o iris.png
from sklearn import tree

with open('iris.dot', 'w') as writer:
    tree.export_graphviz(decision_tree=algo, out_file=writer)

# sys.exit()
# 方式二：直接使用pydotpuls库将数据输出为图像
import pydotplus
from sklearn import tree
# import os
# os.environ["PATH"] += os.pathsep + 'G:/program_files/graphviz/bin'

dot_data = tree.export_graphviz(decision_tree=algo, out_file=None,
                                feature_names=['A', 'B', 'C', 'D'],
                                class_names=['1', '2', '3'],
                                filled=True, rounded=True,
                                special_characters=True
                                )
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("iris0.png")
graph.write_pdf("iris0.pdf")
