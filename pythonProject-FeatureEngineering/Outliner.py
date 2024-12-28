import missingno
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
import sklearn.metrics
from numpy.ma.core import outer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.max_rows', None)     # Tüm satırları göster
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # Float sayıları iki ondalık basamakla göster
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv('datasets/application_train.csv')
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv('datasets/titanic.csv')
    return data

df = pd.read_csv('datasets/titanic.csv')
df.head()

####
#Grafik Teknikle Aykırı Değerler
####

sns.boxplot(x=df["Age"])
plt.show()

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)

iqr = q3 - q1

up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)].index

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)
df[~(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[(df["Age"] < low)].any(axis=None)
####
#Fonksiyonlaştırma
####

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)].head()
df[(df["Fare"] < low) | (df["Fare"] > up)].index

df = pd.read_csv('datasets/titanic.csv')
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if (dataframe[col_name] > up_limit).any() or (dataframe[col_name] < low_limit).any():
        return True
    else:
        return False

check_outlier(df, "Fare")

#####
#grab_col_names
#####



dff = load_application_train()
dff.head()


def grab_col_names(dataframe, cat_th = 10, car_th = 20):#10 ve 20 değerleri yorumu yapan kişiye bırakılmıştır.

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == '0']

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "0"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > cat_th and
                   dataframe[col].dtypes == "0"]

    cat_cols = num_but_cat + cat_cols

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "0"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations:  {dataframe.shape[0]}")
    print(f"Variables:  {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)


####
#Aykırı Değerlerin kendisine ulaşmak için kullanılan fonksiyon
####

def grab_outlier(dataframe, col_name, index=False): # indexi true yaparak aykırı değerleri getirtebiliriz
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10: #shape 0.elamanında gölem sayısı vardır
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outer_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outer_index

grab_outlier(df, "Age") #yaş deeğişkeninde 10dan fazla aykırı değer oldupu için head() getirdi

low, up = outlier_thresholds(df, "Fare")
df.shape
df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

def remove_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
df.shape

df = pd.read_csv('datasets/titanic.csv')
df.head()

for col in num_cols:
    nw_df = remove_outliers(df, col)

df.shape[0] - nw_df.shape[0]


####
#Baskılama Yöntemi (re-assigment with thresholders)
####

low, up = outlier_thresholds(df, "Fare")
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up
df.loc[(df["Fare"] < low), "Fare"] = low

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = pd.read_csv('datasets/titanic.csv')
df.shape

####
#Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
####

#LOF yöntemi lokal yoğunluğu ilgili noktadan uzak olup olmadığına bakılır

#çok değişkenli verileri iki boyuta indirgemek için PCA(temel bileşen analizi) yöntemiyl yapabiliriz.
#100 değişkenin büyük çoğunluğunun taşıdığı bilginin büyük bir çoğunluğunu taşıdığını varsaydığımız iki bileşen aracılığıyla veri temsizl edilebilir.

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape
for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20) #Localoutlier metodunda ön tanımlı olarak verilen 20yi kullanıyoruz
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
#df_scores = -df_scores
#Değerler 1'den uzaklaşınca aykırı değerlere yaklaşır deriz.

np.sort(df_scores)[0:5]

#elbow(dirsek yöntemi)
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()#en dik geçiş noktasını eşik değer larak belirleyebilirim


th = np.sort(df_scores)[3]

df[df_scores < th]
df[df_scores < th].shape

#neden aykırı olduğunu bulmak için bir kaç işlem yapalım

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index) #sildik ancak baskılama işlemini de incelemiz lazım
#hangi değişkene göre baskılamamız gerekiyor.
#ağaç yöntemi kullanıyorsak aykırı veriye dokunmamak daha sağlıklı olabiliyor.
