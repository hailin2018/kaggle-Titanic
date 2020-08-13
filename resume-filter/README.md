# Practice-Project-resume-filter
数据源：
employees_dataset.csv
employees_dataset.csv 为了某公司部分员工的简历信息，包含了：学历（degree）, 毕业院校（education）, 技能（skills）, 曾经工作过的公司（working_experience）,和当前职位（position）几部分信息。
现在的职位有三种：dev（开发工程师），qa（测试工程师）和 manager（经理）
要求：
用本题提供的数据集训练一个模型，用来过滤求职者的简历；
用来判断求职者是：不能被录取、适合当开发工程师、适合当测试工程师，还是适合当经理
验证标准：精准率，召回率和 F1Score

问题分析：
1.分类问题，考虑使用决策树、LR、贝叶斯模型
2.特征预处理：特征degree和职位position是非数值数据，用LabelEncoder将类别特征转为离散值
3.文本特征提取：特征education、skills、working_experience都是文本特征，用词袋模型提取文本特征，
4.数据预处理：OneHotEncoder将离散值特征进行OneHot编码
5.简单交叉验证：按8：2的比例将特征数据集分为训练集和测试集
