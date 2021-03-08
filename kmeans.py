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
dataList = data['TEMP'].values.tolist()

'''
dataList.insert(30, 32)
dataList.insert(50, 9)
dataList.insert(100, 32.2)
dataList.insert(200, 15.2)
dataList.insert(216, 16)
dataList.insert(350, 15)
dataList.insert(351, 32)
dataList.insert(476, 30)
dataList.insert(498, 16)
dataList.insert(295, 30)
'''



data = pd.DataFrame(dataList, columns = ['TEMP'])



data = data.to_numpy()
temp = data


 

plt.plot(data)
plt.show()


data = scale(data)

kmeans = KMeans(n_clusters = 1).fit(data)



center = kmeans.cluster_centers_


distance = sqrt((data - center)**2)

order_index = argsort(distance, axis = 0)
indexes = order_index[-20:]

values = temp[indexes]


plt.plot(temp)
plt.scatter(indexes, values, color='r')
plt.show()


print(indexes)


for i in indexes:
    print(temp[i])



print("Process finished --- %s seconds ---" % (time.time() - start_time))



#points:


