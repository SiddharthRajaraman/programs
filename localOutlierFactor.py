# data preparation
import pandas as pd
import numpy as np
# data visualzation
import matplotlib.pyplot as plt
import seaborn as sns
# outlier/anomaly detection
from sklearn.neighbors import LocalOutlierFactor
import time

start_time = time.time()

df = pd.read_csv('waterquality.csv', encoding = 'latin1')

df = df.dropna(subset = ['TEMP'])
df = df['pH']
df = df.to_frame()



model1 = LocalOutlierFactor(n_neighbors = 23, metric = "minkowski", contamination = 0.03)

y_pred = model1.fit_predict(df)


outlier_index = np.where(y_pred == -1) 

outlier_values = df.iloc[outlier_index]




plt.scatter(range(len(df)),df, color = "b")
plt.scatter(outlier_index, outlier_values, color = "r")
plt.show()

print(outlier_values)

print("Process finished --- %s seconds ---" % (time.time() - start_time))
