import scipy.cluster.hierarchy as shc
from scipy.cluster import hierarchy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cust_df = pd.read_csv("Cust_Segmentation.csv")
df = cust_df.drop('Address', axis=1)
x = df.values[:, 1:]
x = np.nan_to_num(x)
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(x)
labels = k_means.labels_
df["Cluster"] = labels
print(df.groupby("Cluster").mean())

X = df.iloc[:, [1, 3]].values
data_l = hierarchy.linkage(X, method='ward')
dend = hierarchy.dendrogram(data_l)
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

area = np.pi * (x[:, 1])**2
plt.scatter(x[:, 0], x[:, 3], s=area, c=labels.astype(np.float64), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

