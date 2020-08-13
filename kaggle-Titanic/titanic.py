#导入相关库
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pandas import Series,DataFrame


#读取数据
data_train = pd.read_csv(r"D:\GoogleDownload\titanic\train.csv")
#print(data_train.info())  #查看是否有缺失值：Age缺180个、Cabin缺600多、Embarked缺2个
#print(data_train.describe())


'''1.特征工程——特征选择:过滤法'''

##卡方检验选择特征？？？



'''2.特征工程——特征表示'''
#1.缺失值的处理

#Age的缺失值和Cabin的缺失值处理：

from sklearn.ensemble import RandomForestRegressor
 
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
    #predictedAges = rfr.predict(unknown_age[:, 1::])
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

#print(data_train.info())  #Embarked的2个缺失值没有处理


#2.离散特征的连续处理：独热编码
#使用pandas的"get_dummies"来完成这个工作，并拼接在原来的"data_train"之上，如下所示

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
#Embarked和船舱等级没有必要关系，试着把这个特征拿掉，再看看效果

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)     		#有Embarked
#df = pd.concat([data_train, dummies_Cabin, dummies_Sex, dummies_Pclass], axis=1)							#没有Embarked
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

print(df.shape)    #(891, 16)
#print(df.info())   #Embarked的2个缺失值独热码为(0,0,0)


#3.Age的离散化处理:
#年龄段划分

df['Age_clean']=np.where(df['Age']<7,0,
				np.where(df['Age']<19,1,
				np.where(df['Age']<31,2,
				np.where(df['Age']<51,3,4
				))))

#print(df.info())

dummies_Age = pd.get_dummies(df['Age_clean'], prefix= 'Age_clean')
df = pd.concat([df, dummies_Age], axis=1) 
df.drop(['Age', 'Age_clean'], axis=1, inplace=True)

#print(df.info())




'''3.特征工程——数据预处理'''
#1.Age和Fare两个特征的范围和其他特征范围不一致，会影响收敛速度，所以进行预处理
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

#age_scale_param = scaler.fit(df[['Age']])
#df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)

fare_scale_param = scaler.fit(df[['Fare']])
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], fare_scale_param)
#print(df[['Age_scaled','Fare_scaled']])
#print(df.shape)   #(891, 18)


#2.用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')	#有Embarked
#train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*') 	#没有Embarked
train_np = train_df.as_matrix()
#print(train_df.shape) #(891, 15)  乘客id/Age/Fare三个被删除


'''4.简单交叉验证:按7：3的比例将特征数据集分为训练集和测试集'''
dataset_x=train_df.drop(["Survived"],axis=1)
dataset_y=train_df["Survived"]
#print(dataset_x)
#print(dataset_y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset_x,dataset_y,test_size=0.4)
#print(y_test)  #y_test是dataframe


'''5.搭建模型拟合'''
#1.LR模型
from sklearn import linear_model
clf_lr = linear_model.LogisticRegression(C=0.8, penalty='l1', tol=1e-6)  #LR的调参？？？
clf_lr.fit(x_train, y_train)
y_pre_lr = clf_lr.predict(x_test)
#print(y_test_pre)   #y_test_pre是matrix

#2.DT模型
from sklearn import tree
clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(x_train,y_train)
y_pred_dt = clf_dt.predict(x_test)#预测测试数据

#3.SVM模型
from sklearn.svm import SVC
clf_svm = SVC(kernel='rbf',C=20.0)
clf_svm.fit(x_train,y_train)
y_pred_svm = clf_svm.predict(x_test)

#4.GBDT模型
from sklearn.ensemble import GradientBoostingClassifier
clf_gbdt=GradientBoostingClassifier(n_estimators=400,max_depth=30,min_samples_leaf=7, learning_rate=0.3)
clf_gbdt.fit(x_train, y_train)
y_pred_gbdt = clf_gbdt.predict(x_test)

#5.RF模型
from sklearn.ensemble import RandomForestClassifier	#导入RF
clf_rf=RandomForestClassifier(oob_score=True, random_state=10,n_estimators=600,max_depth=10,min_samples_split=6)
clf_rf.fit(x_train, y_train)
y_pred_rf = clf_rf.predict(x_test)



'''6.计算预测准确率'''
preAcc_lr = y_test[y_test==y_pre_lr].size*100/y_test.size           #numpy.array的布尔值索引，将ture的数组合成一个list返回
print("LR模型的准确率为%.2f%%"%preAcc_lr)

preAcc_dt = y_test[y_test==y_pred_dt].size*100/y_test.size           #numpy.array的布尔值索引，将ture的数组合成一个list返回
print("DT模型的准确率为%.2f%%"%preAcc_dt)

preAcc_svm = y_test[y_test==y_pred_svm].size*100/y_test.size           #numpy.array的布尔值索引，将ture的数组合成一个list返回
print("SVM模型的准确率为%.2f%%"%preAcc_svm)

preAcc_gbdt = y_test[y_test==y_pred_gbdt].size*100/y_test.size           #numpy.array的布尔值索引，将ture的数组合成一个list返回
print("GBDT模型的准确率为%.2f%%"%preAcc_gbdt)

preAcc_rf = y_test[y_test==y_pred_rf].size*100/y_test.size           #numpy.array的布尔值索引，将ture的数组合成一个list返回
print("RF模型的准确率为%.2f%%"%preAcc_rf)




y_pred_sum = y_pre_lr + y_pred_dt + y_pred_svm + y_pred_gbdt + y_pred_rf
for i in range(len(y_pred_sum)):
	if y_pred_sum[i]>2:
		y_pred_sum[i] = 1
	else:
		y_pred_sum[i] = 0


preAcc_sum = y_test[y_test==y_pred_sum].size*100/y_test.size           #numpy.array的布尔值索引，将ture的数组合成一个list返回
print("\n集成模型的准确率为%.2f%%"%preAcc_sum)



"""7.分析预测结果、改进模型"""
##查看model各个特征的系数
#w_lr = pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf_lr.coef_.T)})
#print(w)

#查看误分类的数据
origin_data_train = pd.read_csv(r"D:\GoogleDownload\titanic\train.csv")
#bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[y_test != y_test_pre]['PassengerId'].values)]   #待完善代码



"""5.处理test数据"""

data_test = pd.read_csv(r"D:\GoogleDownload\titanic\test.csv")
#print(data_test.info())     #查看缺失值的情况：Age缺100多、Cabin缺400多、Fare缺1个，但Embarked没有缺失

#1.处理Fare的缺失：
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
#2.处理Age的缺失：用同样的RF模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
#3.处理Cabin的缺失:
data_test = set_Cabin_type(data_test)
#4.离散特征的连续处理：独热编码
#使用pandas的"get_dummies"来完成这个工作，并拼接在原来的"data_train"之上，如下所示
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


#5.Age的离散化处理：
df_test['Age_clean']=np.where(df_test['Age']<7,0,
				np.where(df_test['Age']<19,1,
				np.where(df_test['Age']<31,2,
				np.where(df_test['Age']<51,3,4
				))))

#print(df.info())

dummies_Age = pd.get_dummies(df_test['Age_clean'], prefix= 'Age_clean')
df_test = pd.concat([df_test, dummies_Age], axis=1) 
df_test.drop(['Age', 'Age_clean'], axis=1, inplace=True)

print(df_test.info())


#5.特征工程——数据预处理:
#Age和Fare两个特征的范围和其他特征范围不一致，会影响收敛速度，所以进行预处理
#df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']], age_scale_param)                       #当Age离散化时，就不用预处理了
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']], fare_scale_param)
print(df_test.shape)
#6.用正则取出我们要的属性值
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
print(test.shape)



"""6.预测test数据"""

pre_lr = clf_lr.predict(test)
pre_dt = clf_dt.predict(test)
pre_svm = clf_svm.predict(test)

pre_gbdt = clf_gbdt.predict(test)
pre_rf = clf_rf.predict(test)


pre_sum = pre_lr + pre_dt + pre_svm + pre_gbdt + pre_rf
print(pre_sum)

for i in range(len(pre_sum)):
	if pre_sum[i]>2:
		pre_sum[i] = 1
	else:
		pre_sum[i] = 0

print(pre_sum)



result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':pre_sum.astype(np.int32)})
#result.to_csv(r"D:\GoogleDownload\titanic\LR_predictions.csv", index=False)
result.to_csv(r"D:\GoogleDownload\titanic\ensemble_predictions.csv", index=False)







































