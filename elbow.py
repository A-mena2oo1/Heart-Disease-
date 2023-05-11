import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
data = pd.read_csv("Heart_Disease_Prediction.csv")
x = data.iloc[:,[4,7]]
wcss=[]
for i in range(1,10):
 kmeans = KMeans(i)
 kmeans.fit(x)
 wcss.append(kmeans.inertia_)

number_clusters = range(1,10)
plt.plot(number_clusters,wcss,'bx-')
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
