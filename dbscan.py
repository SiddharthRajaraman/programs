import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time


start_time = time.time()

df = pd.read_csv("waterquality.csv", encoding = 'latin1')
print(df.head())


data = df[["pH"]]

model = DBSCAN(eps = 0.9, min_samples = 4).fit(data)

colors = model.labels_
plt.scatter(range(len(data['pH'])),data["pH"] , c = colors)
plt.show()


outliers = data[model.labels_ == -1]
print(outliers)


print("Process finished --- %s seconds ---" % (time.time() - start_time))

