####
#VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN
####


#MATPLOTLIB
#Kategorik değişken: stün grafik, counplot bar
#sayısla değişken: histogram, boxplot


####
#Kategorik Değişken Görselleştirme
####

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show()


####
#Sayısal DEğişken Görselleştirme
####

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

####
#Matplolib'in Özellikeleri
####

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#plot

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

#marker

y = np.array([13, 28, 11, 100])
plt.plot(y, marker='o')
plt.show()

markers = ['o', '*', ',', '.', 'x','+']

#line

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dotted", color = "red") #dashdot
plt.show()


#Multiple Lines
x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

#Labels

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)

plt.title("Bu Ana Başlık")
plt.xlabel("X ekseninin isimlendirilmesi")
plt.ylabel("Y ekseninin isimlendirilmesi")
plt.grid(True)
plt.show()

#Subplots

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)

x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)

plt.show()

#SEABORN
import pandas as pd
import seaborn as pd
from matplotlib import pyplot as plt
df = sns.load_dataset('tips')
df.head()
df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

#sayısal değişkenleri seabornla görselleştirme

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()