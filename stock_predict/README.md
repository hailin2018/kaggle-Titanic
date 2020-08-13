# Practice-Project_stock-predict

数据源：
msft_stockprices_dataset.csv
msft_stockprices_dataset.csv 记录了从2014年9月9日到2018年9月10日的微软股价，包括每日最高价（High Price），最低价（Low Price），开盘价（Open Price），收盘价（Close Price）和交易量（Volume）
要求：
用本题提供的数据集训练一个模型，用来预测微软公司股票的收盘价。
验证标准：
预测准确率=（预测正确样本数）/（总测试样本数）* 100%
可以人工指定一个 ErrorTolerance（一般是10%或者5%），
当 |预测值-真实值| / 真实值 <= ErrorTolerance 时，
我们认为预测正确，否则为预测错误。

问题分析：
1.回归问题，考虑使用线性回归、RF、GBDT三种模型
2.特征volume取值范围和其他特征相差太大，采用归一化预处理
3.交叉验证：简单交叉验证/k折交叉验证
4.ErrorTolerance=[0.1,0.05,0.01]
