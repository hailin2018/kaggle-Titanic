#股票预测器


#导入要用到的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


#读入数据
stockprices_dataset=pd.read_csv('msft_stockprices_dataset.csv')
#dataset_x=stockprices_dataset["High Price", "Low Price", "Open Price", "Volume",axis=1]   #X = np.array(data[used_features])
dataset_x=stockprices_dataset.drop(["Date","Close Price"],axis=1)
dataset_y=stockprices_dataset["Close Price"]


#数据预处理:归一化
from sklearn import preprocessing
dataset_x=preprocessing.MinMaxScaler().fit_transform(dataset_x)


#简单交叉验证：按8：2的比例将特征数据集分为训练集和测试集			NOTE:如果考虑用K折交叉验证该怎么做？？？
x_train,x_test,y_train,y_test=train_test_split(dataset_x,dataset_y,test_size=0.2)


#创建线性回归模型
from sklearn import linear_model
regr=linear_model.LinearRegression()

from sklearn.ensemble import RandomForestRegressor	#导入RF
rf=RandomForestRegressor(oob_score=True, random_state=10,n_estimators=600,max_depth=10,min_samples_split=6)					#定义模型rf0

from sklearn import ensemble
gbdt=ensemble.GradientBoostingRegressor(n_estimators=200,max_depth=3,min_samples_leaf=7, learning_rate=0.12,loss='ls')


#用训练集训练模型
regr.fit(x_train, y_train)
rf.fit(x_train, y_train)
gbdt.fit(x_train, y_train)


#用训练得到的模型预测测试集的数据
y_pred_regr=regr.predict(x_test)
y_pred_rf=rf.predict(x_test)
y_pred_gbdt=gbdt.predict(x_test)


#计算三个模型的准确率
ErrorTolerance=[0.1,0.05,0.01]		#0.1
total=y_test.size			#y_test的数量

error_regr=abs((y_pred_regr-y_test)/y_test)		#计算模型的预测值与实际值的误差
error_rf=abs((y_pred_rf-y_test)/y_test)
error_gbdt=abs((y_pred_gbdt-y_test)/y_test)

for i in range(0,3):

	preAcc_regr=error_regr[error_regr<=ErrorTolerance[i]].size*100/total		#numpy.array的布尔值索引，将ture的数组合成一个list返回
	preAcc_rf=error_rf[error_rf<=ErrorTolerance[i]].size*100/total
	preAcc_gbdt=error_gbdt[error_gbdt<=ErrorTolerance[i]].size*100/total
	
	print("当容错率为%s时："%ErrorTolerance[i])
	print("线性回归模型的准确率为%.2f%%，RF模型的准确率为%.2f%%，GBDT模型的准确率为%.2f%%"%(preAcc_regr,preAcc_rf,preAcc_gbdt))
	print("\n")
	
