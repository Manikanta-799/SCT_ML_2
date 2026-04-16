import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Income':[15,16,17,18,19,40,42,43,44,45],
    'Spending':[39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3)

df['Cluster'] = kmeans.fit_predict(df)

plt.scatter(df['Income'],df['Spending'],c=df['Cluster'])

plt.xlabel("Income")
plt.ylabel("Spending Score")

plt.show()
