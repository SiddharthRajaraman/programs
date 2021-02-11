import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

df = pd.read_csv("waterquality.csv", encoding = 'latin1')
print(df.head())

# input data
data = df[["pH"]]
# specify & fit model
model = DBSCAN(eps = 0.4, min_samples = 10).fit(data)

colors = model.labels_
plt.scatter(range(len(data['pH'])),data["pH"] , c = colors)
plt.show()

# outliers dataframe
outliers = data[model.labels_ == -1]
print(outliers)
'''
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from numpy import random, where
import matplotlib.pyplot as plt

random.seed(7)
x, _ = make_blobs(n_samples=200, centers=1, cluster_std=.3, center_box=(20, 5))

plt.scatter(x[:,0], x[:,1])
plt.show()

dbscan = DBSCAN(eps = 0.28, min_samples = 20)
print(dbscan)

pred = dbscan.fit_predict(x)
anom_index = where(pred == -1)
values = x[anom_index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='r')
plt.show()
'''