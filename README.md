# kaggle-Titanic
1.特征选择：删除name、ticket俩个无关特性，剩下的特征保留
2.缺失值处理：Age用RF填充，Cabin转化为二值特征
3.特征表示：非数值型的离散特征用one-hot编码，即：Cabin、Embarked、Sex、Pclass四个特征用one-hot编码
4.特征预处理：Age和Fare两个特征的范围和其他特征范围不一致，会影响收敛速度，所以进行预处理
5.处理test数据：用同样的方式处理test数据
6.模型融合：选用LR+SVM+DT+GBDT+RF进行模型融合
