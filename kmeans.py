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

data = pd.read_csv('waterquality.csv', encoding = 'latin1', usecols = ['pH'])
#print(data)



data = data.to_numpy()




#print(x)
 
'''
plt.plot(data)
plt.show()
'''

data = scale(data)

kmeans = KMeans(n_clusters = 1).fit(data)
#print(kmeans)


center = kmeans.cluster_centers_
#print(center)

distance = sqrt((data - center)**2)

order_index = argsort(distance, axis = 0)
indexes = order_index[-10:]

values = data[indexes]

'''
plt.plot(data)
plt.scatter(indexes, values, color='r')
plt.show()
'''

print(indexes)
print(values)

print("Process finished --- %s seconds ---" % (time.time() - start_time))