#! usr/bin/python
#coding=utf-8
'''
Created on 2017年9月26日

@author: lianglong
'''

# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # 允许为DataFrame使用display()

# 导入附加的可视化代码visuals.py
# import visuals as vs
# 为notebook提供更加漂亮的可视化
# %matplotlib inline

# 导入人口普查数据
data = pd.read_csv("census.csv")

n_records = data.shape[0]
# 成功 - 显示第一条记录
# display(data.head(n=2))

n_records = data.shape[0]

# TODO：被调查者的收入大于$50,000的人数
n_greater_50k = data[data['income'] == '>50K'].shape[0]

# TODO：被调查者的收入最多为$50,000的人数
n_at_most_50k = data[data['income'] == '<=50K'].shape[0]

# TODO：被调查者收入大于$50,000所占的比例
greater_percent = float(n_greater_50k)/float(n_records)


# 打印结果
# print "Total number of records: {}".format(n_records)
# print "Individuals making more than $50,000: {}".format(n_greater_50k)
# print "Individuals making at most $50,000: {}".format(n_at_most_50k)
# print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

from sklearn.preprocessing import MinMaxScaler

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 显示一个经过缩放的样例记录
# display(features_raw.head(n = 1))

# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)
# print features

# TODO：将'income_raw'编码成数字值
income = income_raw.map({'<=50K':0,'>50K':1})
# print income

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
# print "{} total features after one-hot encoding.".format(len(encoded))

# 移除下面一行的注释以观察编码的特征名字
# print encoded

# 导入 train_test_split
from sklearn.model_selection import train_test_split

# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0,
                                                    stratify = income)
# 将'X_train'和'y_train'进一步切分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                    stratify = y_train)

# 显示切分的结果
# print "Training set has {} samples.".format(X_train.shape[0])
# print "Validation set has {} samples.".format(X_val.shape[0])
# print "Testing set has {} samples.".format(X_test.shape[0])


#如果我们选择一个无论什么情况都预测被调查者年收入大于 $50,000 的模型，那么这个模型在验证集上的准确率，查准率，查全率和 F-score是多少？
#不能使用scikit-learn，你需要根据公式自己实现相关计算。

#TODO： 计算准确率
accuracy = y_val[y_val == 1].shape[0]/float(y_val.shape[0])

# TODO： 计算查准率 Precision
precision = y_val[y_val == 1].shape[0]/float(y_val.shape[0])

# TODO： 计算查全率 Recall
recall = y_val[y_val == 1].shape[0]/float(y_val[y_val == 1].shape[0])

# TODO： 使用上面的公式，设置beta=0.5，计算F-score
fscore = (1 + .5*.5) * (precision*recall/(.5*.5*precision+recall))

# 打印结果
print "Naive Predictor on validation data: \n \
    Accuracy score: {:.4f} \n \
    Precision: {:.4f} \n \
    Recall: {:.4f} \n \
    F-score: {:.4f}".format(accuracy, precision, recall, fscore)

# TODO：从sklearn中导入两个评价指标 - fbeta_score和accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_val, y_val): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_val: features validation set
       - y_val: income validation set
    '''
    
    results = {}
    
    # TODO：使用sample_size大小的训练数据来拟合学习器
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # 获得程序开始时间
    learner = learner.fit(X_train.head(sample_size), y_train.head(sample_size))
    end = time() # 获得程序结束时间
    
    # TODO：计算训练时间
    results['train_time'] = end - start
    
    # TODO: 得到在验证集上的预测值
    #       然后得到对前300个训练数据的预测结果
    start = time() # 获得程序开始时间
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train.head(300))
    end = time() # 获得程序结束时间
    
    # TODO：计算预测用时
    results['pred_time'] = end - start
            
    # TODO：计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(predictions_train, y_train.head(300))
        
    # TODO：计算在验证上的准确率
    results['acc_val'] = accuracy_score(predictions_val, y_val)
    
    # TODO：计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train.head(300), predictions_train, average='macro', beta=0.5)
        
    # TODO：计算验证集上的F-score
    results['f_val'] = fbeta_score(y_val, predictions_val, average='macro', beta=0.5)
       
    # 成功
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # 返回结果
    return results

# TODO：从sklearn中导入三个监督学习模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  

# TODO：初始化三个模型
clf_A = DecisionTreeClassifier(random_state=0)
clf_B = SVC(random_state=0)
clf_C = LogisticRegression(random_state=0)  

# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = len(y_train)/100
samples_10 = len(y_train)/10
samples_100 = len(y_train)

# 收集学习器的结果
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)

# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
# from sklearn.metrics.scorer import make_scorer

from sklearn.cross_validation import KFold
from sklearn.metrics import make_scorer
from sklearn import grid_search
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
# TODO：初始化分类器
clf = GradientBoostingClassifier(
                loss='deviance', ##损失函数默认deviance  deviance具有概率输出的分类的偏差
                learning_rate=0.1, # 默认0.1学习率
                n_estimators=100, # 迭代训练次数，回归树个数（弱学习器）
                max_depth=3, # 默认值3最大树深度
#                 subsample=0.7, # 子采样率，为1表示全部，推荐0.5-0.8默认1
#                 criterion='friedman_mse', # 判断节点是否继续分裂采用的计算方法
#                 min_samples_split=300, # 生成子节点所需要的最小样本数，浮点数代表百分比
#                 min_samples_leaf=100, # 叶子节点所需要的最小样本数，浮点数代表百分比
#                 min_weight_fraction_leaf=0.,
#                 min_impurity_split=1e-7, # 停止分裂叶子节点的阀值
                init=None,
                random_state=0, # 随机种子，方便重现，如review结果，调参优化能确定是参数调整的结果还是random_state的波动
                #max_features=50, # 寻找最佳分割点要考虑的特征数量auto和None全选/sqrt开方/log2对数/int自定义个数/float百分比
                verbose=0,
                max_leaf_nodes=None, # 叶子节点的个数，None不限数量
                warm_start=False, # True在前面基础上增量训练（重设参数减少训练次数） False默认值擦除重新训练
                presort='auto')  

# TODO：创建你希望调节的参数列表
parameters = {"max_depth":[3, 5, 7], "n_estimators":[100,300,500]}

# TODO：创建一个fbeta_score打分对象

def performance_metric(y_test, y_pred):
    return fbeta_score(y_test, y_pred, beta = 0.5)
scorer = make_scorer(performance_metric)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = grid_search.GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=1, scoring=scorer, verbose=10)# cv=cross_validator, 

# TODO：用训练数据拟合网格搜索对象并找到最佳参数
grid_obj.fit(X_train, y_train)
# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
predictions = (clf.fit(X_train, y_train)).predict(X_val)
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
