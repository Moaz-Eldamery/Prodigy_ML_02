
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from matplotlib import pyplot as plt

# read the csv file
data = pd.read_csv('project_02\\archive\\Mall_Customers.csv')
data

# removing unused data and fitting it to clustering
df = data
df = df.drop(columns=['CustomerID','Gender'])
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
print(df.head())

# find obtimal K cluster number by using the elbow method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# display the result of the elbow method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# apply k-means clustering with the chosen number of clusters (e.g., k=4)
# transforming the data into 2 components to fit the clusterign algorithm using PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

k = 4  
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_pca)

# Add cluster labels to the original dataset
data['Cluster'] = clusters

# display the first few rows of the dataset with cluster labels
print(data.head())

# visualize clusters
plt.figure(figsize=(8, 5))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

