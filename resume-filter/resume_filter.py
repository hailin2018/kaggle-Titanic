#简历筛选器


#导入要用到的库
import pandas as pd
import numpy as np


#读取数据
dataset = pd.read_csv('./employees_dataset.csv')


#数据预处理：LabelEncoder将类别特征转为离散值
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y=le.fit_transform(dataset['position'].values)

used_features = ["degree"]
X = dataset[used_features].values
X[:,0]=le.fit_transform(X[:,0])

print(X.shape)


#文本特征提取：用词袋模型提取文本特征，
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X=np.concatenate((X, vectorizer.fit_transform(dataset['education'].values).todense()), axis=1)
X=np.concatenate((X, vectorizer.fit_transform(dataset['skills'].values).todense()), axis=1)
X=np.concatenate((X, vectorizer.fit_transform(dataset['working_experience'].values).todense()), axis=1)

print(X.shape)


#数据预处理：OneHotEncoder将离散值特征进行OneHot编码
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
enc.fit(X)
X=enc.transform(X).toarray()

print(X.shape)


#简单交叉验证：按8：2的比例将特征数据集分为训练集和测试集			NOTE:如果考虑用K折交叉验证该怎么做？？？
from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)# 按分割测试数据与训练数据
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)# 按分割测试数据与训练数据


#训练模型
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,Y_train)

y_pred = gnb.predict(X_test)#预测测试数据

print("测试集总样本数：{}  误分类数：{}  NB的准确率：{:05.2f}%"
          .format(
              X_test.shape[0],
              (Y_test != y_pred).sum(),
              100*(1-(Y_test != y_pred).sum()/X_test.shape[0])
    ))
	

from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(C=1e5)  
lr.fit(X_train,Y_train)
y_pred = lr.predict(X_test)#预测测试数据
print("测试集总样本数：{}  误分类数：{}  LR的准确率：{:05.2f}%"
          .format(
              X_test.shape[0],
              (Y_test != y_pred).sum(),
              100*(1-(Y_test != y_pred).sum()/X_test.shape[0])
    ))
	
	
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)#预测测试数据
print("测试集总样本数：{}  误分类数：{}  DT的准确率：{:05.2f}%"
          .format(
              X_test.shape[0],
              (Y_test != y_pred).sum(),
              100*(1-(Y_test != y_pred).sum()/X_test.shape[0])
    ))

	
	

