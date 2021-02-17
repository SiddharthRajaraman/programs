import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time

#compile time stuff
start_time = time.time()

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


print("Process finished --- %s seconds ---" % (time.time() - start_time))

