#####K-Ortalamalar(K-Means)

#1.Adım: Küme sayısı belirlenir
#2.Adım: Rastgele k merkez seçilir
#3.Adım: Her gözlem için k merkezlere uzaklık hesaplanır
#4.Adım: Her gözlem en yakın olduğu merkeze atanır
#5.Adım: Atama işlemlerinden sonra oluşan kümeler için tekrar merkez hesaplamaları yapılır
#6.Adım: Bu işlem belirli bir iterasyınca tekrar edilir ve küme içi hata karaeler toplamının minimum olduğu durumdaki gözlemlerin kümelenme yapısı nihai kümelenme olarak seçilir

from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


######K-Means

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

df.head()
df.isnull().sum()
df.info()
df.describe().T

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

kmeans = KMeans(n_clusters=5, random_state=17).fit(df)
pprint(kmeans.get_params())

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

####Optimum Küme Sayısının Belirlenmesi

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_
#6

#####Final Cluster'ların Oluşturulması

kmeans = KMeans(n_clusters=5, random_state=17).fit(df)
kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters = kmeans.labels_

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df["cluster"] = clusters
df["cluster"] = df["cluster"] + 1
df.head()

df[df["cluster"]==1] #1.clusterdaki bilgiler

df.groupby("cluster").agg(["count", "mean", "median"])

df.to_csv("clusters.csv")


####Hiyerarşik Kümeleme Analizi
#Divisive(Bölümleyici) ve Agglomerative(Birleştirici) şekilde ikiye ayrılır
#Kmeansten farklı küme oluşturma sürecine dışardan müdahale edemşyoruz ancak hiyerarşik kümelemede yorumlamaya göre dışardan müdahale edebiliriz

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")

plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
          leaf_rotation=10)
plt.show()

#####Küme Sayısını Belirlemek

plt.figure(figsize=(7,5))
plt.title("Denrograms")
dend = dendrogram(hc_average,
                  truncate_mode='lastp',
                  p=10,
                  show_leaf_counts=True,
                  leaf_rotation=10)
plt.axhline(y=0.6, color='black', linestyle='--')
plt.axhline(y=0.5, color='blue', linestyle='--')
plt.show()

#Final Modeli Oluşturmak

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage='average')
df = pd.read_csv("datasets/USArrests.csv", index_col=0)
cluster = cluster.fit_predict(df)
df["hi_cluster_no"] = cluster
df["hi_cluster_no"] = df["hi_cluster_no"] + 1

####Principal Component Analysis(PCA)
#Veri setinde ki bilginin daha az veriyle açıklanması

df = pd.read_csv("datasets/hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtype != "O" and "Salary" not in col]

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)#Kümülatif toplama baktık


####Optimum Bileşen Sayısı

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısı")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()

###Final PCA

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)