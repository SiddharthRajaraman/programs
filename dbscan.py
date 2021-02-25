import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import seaborn as sns
sns.set()


start_time = time.time()

df = pd.read_csv("waterquality.csv", encoding = 'latin1')

df = df.dropna(subset = ["TEMP"])


df = df['pH']
df = df.to_frame()

'''
#for eps
neigh = NearestNeighbors(n_neighbors = 2)

nbrs = neigh.fit(df)

distances, indices = nbrs.kneighbors(df)

distances = np.sort(distances, axis = 0)
distances = distances[:,1]
plt.plot(distances)
'''

model = DBSCAN(eps = 0.1, min_samples = 4).fit(df)
#default: eps: 0.4 min_samples: 5
#adjusted Temp: eps --> 0.1 min_samples: 4
#adjusted pH: eps --> 0.1 min_samples: 4



colors = model.labels_

plt.scatter(range(len(df)),df , c = colors)
plt.show()


outliers = df[model.labels_ == -1]
print(outliers)


print("Process finished --- %s seconds ---" % (time.time() - start_time))