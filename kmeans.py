from sklearn.cluster import KMeans
from numpy import sqrt, random, array, argsort
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time

#compile time stuff
start_time = time.time()

data = pd.read_csv('waterquality.csv', encoding = 'latin1')
#print(data)

data = data.dropna(subset = ["TEMP"])


data = data['TEMP']
data = data.to_frame()

data = data.to_numpy()
temp = data



 
'''
plt.plot(data)
plt.show()
'''

data = scale(data)

kmeans = KMeans(n_clusters = 1).fit(data)



center = kmeans.cluster_centers_


distance = sqrt((data - center)**2)

order_index = argsort(distance, axis = 0)
indexes = order_index[-10:]

values = data[indexes]


plt.plot(data)
plt.scatter(indexes, values, color='r')
plt.show()


print(indexes)


for i in indexes:
    print(temp[i])



print("Process finished --- %s seconds ---" % (time.time() - start_time))
