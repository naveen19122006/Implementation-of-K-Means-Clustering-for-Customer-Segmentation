# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries .

2.Read the data frame using pandas.

3.Get the information regarding the null values present in the dataframe.

4.Apply label encoder to the non-numerical column inoreder to convert into numerical values.

5.Determine training and test data set.

6.Apply k means clustering for customer segmentation.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Naveen Kumar E
RegisterNumber:  212224230181
*/
```
### Import Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```
### Load the Dataset
```python
# Load dataset
data = pd.read_csv(r"C:\Users\admin\Downloads\Mall_Customers.csv")

# Display first 5 rows
print(data.head())
```
### Select Features for Clustering
```python
# Selecting relevant features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
```
### Feature Scaling 
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
### Finding Optimal K (Elbow Method)
```python
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```
### Apply K-Means
```python
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
```
### Add Cluster Labels to Dataset
```python
data['Cluster'] = y_kmeans
print(data.head())
```
### Visualising the Clusters
```python
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s=50, label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s=50, label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s=50, label='Cluster 3')
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1], s=50, label='Cluster 4')
plt.scatter(X_scaled[y_kmeans == 4, 0], X_scaled[y_kmeans == 4, 1], s=50, label='Cluster 5')

plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```
## Output:
### data.head()

<img width="577" height="100" alt="image" src="https://github.com/user-attachments/assets/65443645-9c94-4970-a711-57a0cb6f73a6" />

### data.info()

<img width="481" height="214" alt="image" src="https://github.com/user-attachments/assets/49eddcca-ce5b-41f5-90c9-a4926e4f4adc" />

### data.isnull().sum()

<img width="282" height="121" alt="image" src="https://github.com/user-attachments/assets/362762c9-24cd-4806-85d3-6b683e574ed1" />

### Elbow Method

<img width="651" height="447" alt="image" src="https://github.com/user-attachments/assets/95308ce2-9a86-4d7b-968e-8c03ac714742" />

### data['Cluster'] = y_kmeans
### print(data.head())

<img width="744" height="241" alt="image" src="https://github.com/user-attachments/assets/3b24ab4d-aa17-472e-8a76-c61417e0a354" />

### K Means cluster graph

<img width="749" height="546" alt="image" src="https://github.com/user-attachments/assets/58ae61f7-1dff-442f-b921-d010bd75d1ec" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
