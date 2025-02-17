import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('CC GENERAL.csv')
x = dataset.iloc[:, 1:18].values

# Impute missing values using the mean of each column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean of the column
x = imputer.fit_transform(x)

from sklearn.cluster import KMeans
wccs_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wccs_list.append(kmeans.inertia_)
plt.plot(range(1, 11), wccs_list)
plt.title('the elbow method')
plt.xlabel('num of clusters')
plt.ylabel('wccs_list')
plt.show()



kmeans=KMeans(n_clusters=8,init='k-means++',random_state=42)
y_predict=kmeans.fit_predict(x)




plt.scatter(x[y_predict==0,0],x[y_predict==0,1],s=100,c='blue',label='cluster1')
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],s=100,c='green',label='cluster2')
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],s=100,c='red',label='cluster3')
plt.scatter(x[y_predict==3,0],x[y_predict==3,1],s=100,c='cyan',label='cluster4')
plt.scatter(x[y_predict==4,0],x[y_predict==4,1],s=100,c='magenta',label='cluster5')
plt.scatter(x[y_predict==5,0],x[y_predict==5,1],s=100,c='orange',label='cluster6')
plt.scatter(x[y_predict==6,0],x[y_predict==6,1],s=100,c='gray',label='cluster7')
plt.scatter(x[y_predict==7,0],x[y_predict==7,1],s=100,c='pink',label='cluster8')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.title('clusters of customers')
plt.xlabel('features')
plt.ylabel('customers')