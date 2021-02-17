from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import matplotlib.pyplot as plt
import pandas as pd
import time

#compile time stuff
start_time = time.time()

x, _ = make_blobs(n_samples=200, centers=1, cluster_std=.3, center_box=(10,10))

'''
plt.scatter(x[:,0], x[:,1])
plt.show()
'''

lof = LocalOutlierFactor(n_neighbors=20, contamination=.03)

y_pred = lof.fit_predict(x)

lofs_index = where(y_pred==-1)
values = x[lofs_index]

'''
plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0],values[:,1], color='r')
plt.show()
'''

# data preparation
import pandas as pd
import numpy as np
# data visualzation
import matplotlib.pyplot as plt
import seaborn as sns
# outlier/anomaly detection
from sklearn.neighbors import LocalOutlierFactor

# data
#df = pd.DataFrame(np.array([[0,1], [1,1], [1,2], [2,2], [5,6]]), columns = ["x", "y"], index = [0,1,2,3,4])

df = pd.read_csv('waterquality.csv', encoding = 'latin1', usecols = ['pH'])


# plot data points
'''
plt.scatter(range(len(df['pH'])),df["pH"], color = "b", s = 65)
plt.grid()
'''
# model specification
model1 = LocalOutlierFactor(n_neighbors = 10, metric = "minkowski", contamination = 0.02)
# model fitting
y_pred = model1.fit_predict(df)
#print(y_pred)

# filter outlier index
outlier_index = np.where(y_pred == -1) # negative values are outliers and positives inliers
# filter outlier values
outlier_values = df.iloc[outlier_index]
# plot data
'''
plt.scatter(range(len(df['pH'])),df["pH"], color = "b", s = 65)
# plot outlier values
plt.scatter(range(len(outlier_values['pH'])), outlier_values['pH'], color = "r")
plt.show()
'''
print(outlier_values)


print("Process finished --- %s seconds ---" % (time.time() - start_time))



