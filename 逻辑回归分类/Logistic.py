import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # 画图
import copy # 用于拷贝数据副本
import os.path


plt.rcParams['axes.unicode_minus']=False # 显示负数
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文


from sklearn.model_selection import train_test_split # 划分训练集和测试集的包
from sklearn.metrics import confusion_matrix, classification_report # 画混淆矩阵和输出分类评价指标的包
from sklearn.metrics import accuracy_score # 计算精确度的包
import seaborn as sns # 将混淆矩阵可视化的包


def Logistic(x):
    return 1/(1+np.exp(-x))


class Logistic_regression(object):
    def __init__(self, learning=0.01, end=0):
        self.end = end
        self.learning = learning
        self.w = 0
    
    def fit(self, x, y):
        # 初始化参数
        x = np.concatenate((x,np.ones(x.shape[0]).reshape(-1, 1)),axis = 1)
        w = np.ones(x.shape[1])
        
        c = 0
        while 1:
            t = np.zeros(x.shape[1])
            for i in range(x.shape[0]):
                t += (1 - Logistic(y[i]*np.dot(w.T, x[i])))*y[i]*x[i]
            c += 1
            # 迭代的截止条件:1.每次迭代w都不在更新, 即找到了极小值点; 2. 规定迭代次数, 本例规定1000次后不在迭代. 
            if np.all(np.abs(t) <= self.end) or c >= 1000:
                break
            w = w + t
        self.w = w
        return
    
    def predict(self, x):
        x = np.concatenate((x,np.ones(x.shape[0]).reshape(-1, 1)),axis = 1)
        y_pre = Logistic(np.dot(x, self.w.T))
        y_pre[y_pre >= 0.5] = 1
        y_pre[y_pre < 0.5] = -1
        return y_pre


data = pd.read_excel(os.path.dirname(__file__) + "\\data.xls")

classes = data["y"].unique().tolist()

data_c = copy.copy(data)
x = data_c[["x1", "x2"]].values
y = data_c["y"].values
x_train, x_test, y_train_pri, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

lrs = {} # 3个分类器的字典, 将每个分类器和置信度按键值对的形式存入字典

for cls in classes:
    y_train = copy.copy(y_train_pri)
    y_train[y_train != cls] = -1
    y_train[y_train == cls] = 1
    lr = Logistic_regression()
    lr.fit(x_train, y_train)
    y_pre = lr.predict(x_train)
    lrs[lr] = accuracy_score(y_train, y_pre)
    
y_test_pre = []
for key, value in lrs.items():
    y_test_pre.append(key.predict(x_test).tolist())
y_test_pre = np.array(y_test_pre).T

y_pre = []
for row in y_test_pre:
    t = 0
    weight = 0
    if np.all(row == -1):
        t = list(lrs.values()).index(min(lrs.values()))
    for j in range(len(row)):
        if row[j] == -1:
            continue
        elif list(lrs.values())[j]>=weight:
            t = j
            weight = list(lrs.values())[j]
        else:
            continue
    y_pre.append(t)

print(classification_report(y_test, y_pre))

cm = confusion_matrix(y_test, y_pre, labels=[0, 1, 2])

ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels([0, 1, 2])
ax.yaxis.set_ticklabels([0, 1, 2])
plt.show()

plt.figure(figsize=(16, 8), facecolor='pink')
plt.scatter(x_test[:, 0], x_test[:, 0], c=y_pre)
plt.show()