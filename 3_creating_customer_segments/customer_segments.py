#! usr/bin/python
#coding=utf-8
'''
Created on 2017年9月13日 11:00:31

@author: lianglong
'''
# 引入这个项目需要的库
import numpy as np
import pandas as pd
import visuals as vs
from IPython.display import display # 使得我们可以对DataFrame使用display()函数

# 设置以内联的形式显示matplotlib绘制的图片（在notebook中显示更美观）
# %matplotlib inline

# 载入整个客户数据集
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
#     print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    
    
# TODO：从数据集中选择三个你希望抽样的数据点的索引
indices = [1,201,401]

# 为选择的样本建立一个DataFrame
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
# print "Chosen samples of wholesale customers dataset:"
# display(samples)

# 生鲜、牛奶、食品杂货、速冻、卫生纸、熟食
# TODO：为DataFrame创建一个副本，用'drop'函数丢弃一些指定的特征
new_data = data.copy();
y = new_data['Delicatessen'].as_matrix()
new_data.drop(['Delicatessen'], axis = 1, inplace = True)
new_data = new_data.as_matrix()

# TODO：使用给定的特征作为目标，将数据分割成训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, y, test_size = 0.1, random_state=1)


# TODO：创建一个 DecisionTreeRegressor（决策树回归器）并在训练集上训练它
from sklearn.tree import DecisionTreeRegressor 
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# TODO：输出在测试集上的预测得分
score = regressor.score(X_test, y_test)
print score


# TODO：使用自然对数缩放数据
log_data = np.log(data)

# TODO：使用自然对数缩放样本数据
log_samples = np.log(samples)

from collections import Counter
c = Counter()
# 对于每一个特征，找到值异常高或者是异常低的数据点
for feature in log_data.keys():
#     print log_data[feature]
    # TODO：计算给定特征的Q1（数据的25th分位点）
    Q1 = np.percentile(log_data[feature],25)#25%分位数
#     print Q1
    # TODO：计算给定特征的Q3（数据的75th分位点）
    Q3 = np.percentile(log_data[feature],75)
#     print Q3
    # TODO：使用四分位范围计算异常阶（1.5倍的四分位距）
    step = (Q3-Q1)*1.5
#     print step
    # 显示异常点
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    c += Counter(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index)
    
# 可选：选择你希望移除的数据点的索引
outliers = filter(lambda item: c[item]>1,c)

print outliers

# 如果选择了的话，移除异常点
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

# TODO：通过在good data上使用PCA，将其转换成和当前特征数一样多的维度
from sklearn.decomposition import PCA
pca = PCA(n_components=6).fit(good_data)

# TODO：使用上面的PCA拟合将变换施加在log_samples上
pca_samples = pca.transform(log_samples)
# print pca_samples
# 生成PCA的结果图
pca_results = vs.pca_results(good_data, pca)
# print np.round(pca_samples, 4)
# print pca_results.index.values

# display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

# TODO：通过在good data上进行PCA，将其转换成两个维度
pca = PCA(n_components=2).fit(good_data)

# TODO：使用上面训练的PCA将good data进行转换
reduced_data = pca.transform(good_data)

# TODO：使用上面训练的PCA将log_samples进行转换
pca_samples = pca.transform(log_samples)

# 为降维后的数据创建一个DataFrame
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# TODO：在降维后的数据上使用你选择的聚类算法
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusterer = kmeans.fit(reduced_data)

# TODO：预测每一个点的簇
preds = kmeans.predict(reduced_data)

# TODO：找到聚类中心
centers = kmeans.cluster_centers_
# print centers
# TODO：预测在每一个转换后的样本点的类
sample_preds = kmeans.predict(pca_samples)

# TODO：计算选择的类别的平均轮廓系数（mean silhouette coefficient）
from sklearn.metrics import silhouette_score
score = silhouette_score(reduced_data, kmeans.labels_, metric="euclidean")
# print score

# TODO：反向转换中心点
log_centers = pca.inverse_transform(centers)

# TODO：对中心点做指数转换
true_centers = np.exp(log_centers)

# 显示真实的中心点
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

