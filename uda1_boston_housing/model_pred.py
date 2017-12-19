#! usr/bin/python
#coding=utf-8
'''
Created on 2017年9月13日 11:00:31

@author: lianglong
'''

import pandas as pd
import numpy as np
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import r2_score

def get_data():
    # 读取csv文件
    X = pd.read_csv("bj_housing.csv", low_memory=False)
    
    # 提取特征和标签
    y = X['Value'].as_matrix()
    X.drop(['Value'],axis=1,inplace=True)
    X = X.as_matrix()
    # 分隔成训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
    return X_train, X_test, y_train, y_test

def fit_model(X_train, y_train):
    # 创建决策树模型
    from sklearn.tree import DecisionTreeRegressor 
    model = DecisionTreeRegressor()
    
    from sklearn.cross_validation import KFold
    from sklearn.metrics import make_scorer
    from sklearn import grid_search
    from sklearn import metrics
    
    cross_validator = KFold(5)
    param_grid = {"max_depth":[4,5,6,7],
#                   "min_samples_split": [30,20,40],
#                   "min_samples_leaf":[10,20,30]
                  }
    
    from sklearn.metrics import r2_score
    def performance_metric(y_test, y_pred):
        score = r2_score(y_test, y_pred)
        return score
    
    scoring_fnc = make_scorer(performance_metric)
    
    model = grid_search.GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=cross_validator, scoring=scoring_fnc, verbose=10)
    
    
    model.fit(X_train, y_train)
    print(model.best_estimator_)
    print(model.grid_scores_)
    print(model.best_params_)
    print(model.best_score_)
    return model.best_estimator_

X_train, X_test, y_train, y_test = get_data()
model = fit_model(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print "r2: ", r2
