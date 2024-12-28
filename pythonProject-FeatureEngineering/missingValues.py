####
#Eksik Değerler
####
#Silme yaklaşımı, Değer Atama Yöntemleri, Tahmine Dayalı Yöntemler
#Eksik verilerin rastgele ortaya çıkıp çıkmadığı kısmına dikkat etmeeliyiz
from idlelib.editor import darwin
import missingno
import missingno as msno
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = sns.load_dataset('titanic')
df.head()
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)

df.isnull().values.any() #hücrelerde eksiklik var mı?

df.isnull().sum() #Değişkenlerdeki eksik sayıları yaz

df.notnull().sum()#değişkenlerdeki tam değer sayısı

df.isnull().sum().sum() #veri setindeki toplam eksik değer sayısı

df[df.isnull().any(axis=1)] #en az 1 tane eksik değere sahip olan gözlem birimleri

df[df.notnull().any(axis=1)]#tam olan gözlem birimleri

df.isnull().sum().sort_values(ascending=False)#azalan şekilde sıralama

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)#oranını öğrenmek için yapıyoruz

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)
####
#Eksiklileri giderme
####
#ağaca dayalı yöntemler kullanılıyorsa eksiklikler gözardı edilebilir.

#Hızlıcı silmek. eksik olan her satırı siler
df.dropna().shape

#Basit atama yöntemleri
df["age"].fillna(df["age"].mean()).isnull().sum()#yaş değişkenindeki boşlukları, yaş değişkeninin ortalamsını ekle
df["age"].fillna(df["age"].median()).isnull().sum()#medianla doldur
df["age"].fillna(0).isnull().sum()#0 ile doldur

##df.apply(lambda x: x.fillna(x.mean()), axis=0)
#ağaşı doğru satırlara bakmak, ancak veri setinde objeleder olacağı için bu kısım hata veriri
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype in ['float64', 'int64'] else x, axis=0)

df["embarked"].fillna(df["embarked"].mode()[0]).isnull().sum()

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

#Kategorik Değişken Kırılımında Değer Atama

df.groupby("sex")["age"].mean()

df["age"].mean()

df["age"].fillna(df.groupby("sex")["age"].transform("mean")).isnull().sum()
#yaşlardaki boşlukları cinsiyete özel doldurma

df.loc[(df["age"].isnull()) & (df["sex"] == "female"), "age"] = df.groupby("sex")["age"].mean()["female"]
#yaş değişkeninde eksiklik olup kadın olanlar
df.groupby("sex")["age"].mean()["female"]

df.loc[(df["age"].isnull()) & (df["sex"] == "male"), "age"] = df.groupby("sex")["age"].mean()["male"]
df.isnull().sum()

#Tahmine Dayalı Atama ile Doldurma

#eksikliğe sahip olan değişkeni bağımlı değişken diğerlerini bağımsı zdeğişken olarak alacağız

df = sns.load_dataset('titanic')
df.head()

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
num_cols = [col for col in num_cols if col not in "passengerId"]

#Label encoding ve one hot encoding

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

#değişkenlerin standartlaştırılması

scaler =MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

#knn'in uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5) # en yakın 5 komşunun ortalamasıyla boşluğu doldurur
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
#ilgili dönüştürme bilgisini geriye dönüştürüyoruz
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

#Doldurulan değerleri görmek için
df["age_imputed_knn"] = dff[["age"]]
df.loc[df["age"].isnull(), ["age", "age_imputed_knn"]]


#Eksik verilerin gelişmiş analizi

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()#Değişkenlerdeki eksikliklerin birlikte oluşup oluşmadığına bakrız

msno.heatmap(df)
plt.show()#Eksik değerlerin rassallığıyla ilgilenirken bu grafik faydalı oluyor

#Eksik Değerlerin Bağımlı Değişken ile Analizi

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n")


missing_vs_target(df, "survived", na_cols)
#B