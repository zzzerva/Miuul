#Feature Extraction(Özellik Çıkarımı)

####
#Ham veriden değişken üretmek
####

#Yapısal olmayan değişkenleri makinelere aktarırken kullanılabilir


#Binary Features: Flag, Bool, True-False

import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # Float sayıları iki ondalık basamakla göster
pd.set_option('display.width', 500)

def load():
    data = pd.read_csv('datasets/titanic.csv')
    return data

df  = load()
df.head

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int') #dolu olanlara 1 boş olanlara 0 yadık

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})
#Survived için değerlendirdiğimizde kabin numarası olanların hayatta kalma olasılığı fala olduğu için kayda değer bir veri içeriyor


from statsmodels.stats.proportion import proportions_ztest

test_start, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                              df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                       nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_start, pvalue))
#Türetilen değişken anlamlı mı değğil mi diye kontrol ediyoruz

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
#Ailesiyle olmayanların hayatta kalma durumu şeklinde bir değişken oluşturduk
df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_start, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                              df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                       nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_start, pvalue))

df.head()

####
#Letter Count
####

df["NEW_NAME_COUNT"] = df["Name"].str.len()

####
#Word Count
####

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

####
#Özel Yapıları Yakalamak
####

df["NEM_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEM_NAME_DR").agg({"Survived": ["mean", "count"]})


####
#Regex ile Değişken Türetme
####

df.head()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["mean", "count"]})

####
#Date DEğişkenleri Üretmek
####
from datetime import date
dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format='%Y-%m-%d %H:%M:%S')

dff['year'] = dff['Timestamp'].dt.year

dff['month'] = dff['Timestamp'].dt.month

dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year
#month_diff = iki tarih arasındaki ay farkı heabı : yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month

#day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()