######
#PANDAS
######

#PANDAS Series
#Reading Data
#Quick Look at Data
#Selection in Pandas
#Aggregation & Grouping (Toplulaştırma ve Gruplaştırma)
#Apply and Lambda
#Join (Birleştirme İşlemleri)


##PANDAS SERİES
#Tek boyutlu ve index bilgisi barındıran veri tipidir.

import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index #index bbilgisine ulaşmak için kullanılan methood
s.dtype #serinin tip bilgisini öğrenmek için kullanılan method
s.size #eleman sayısını öğrenmek
s.ndim #boyut bilgisi
s.values #içerisindeki değerler
type(s.values) #Pandas serisinin sonuna values ifadesi eklenince indeks bilgisiyle ilgilenilmediği bilgisinden dolayı tip numpy array olarak döner.
s.head(3) #ilk 3 ifadeyi getirir
s.tail(3) #son 3 ifadeyi getirir

###
#Veri Okuma
###

df = pd.read_csv("sample_data.csv") #bir csv dosyasından veri okuma işlemi
df.head() #ctr + sol click dökumanın detaylarına gittik.


##
#Veriye Hızlı Bakış
##

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic") #seaborn kütüphanesindeki titanic verisetini okuduk. Titanic veri seti yolculuk sonrası hayatta kalanlarla ilgili bir veri setidir. Survived değişkeni 0 ise yolcu hayatta kalamamış 1 ie hayatta kalmış demektir.
df.head()
df.tail()
df.shape #891 satır 15 sütun
df.info() #daha detaylı bilgi almak için kullanılır.
#object ve category değişkenlerini genel olarak kategorik değişken olarak adlandırırız.
df.columns #isimlerine erişmek için
df.index #index bilgisine erişmek için
df.describe().T #.T transpozunu almak anlamında kullanılmıştır. Daha okunabilir olması için kullanılır.
df.isnull().values.any() #veri setinde en az bir eksiklik var mıdır
df.isnull().sum() #herbir değişkende kaçtane eksiklik olsuğunu hesapladı.
df["sex"].head() #dataframe içerisinden değişken seçmek içi kullanılır.
df["sex"].value_counts() #istenen değişken türlerini ve her birinden kaçar tane olduğunu belirtir.


###
#Pandas'ta Seçim İşlemleri
###


df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13] #0. indis dahil 13. indis dahil değil.
df.drop(0, axis=0).head() #drop silme işlemi için kulllanılldı. verisetini gözlemlemek için head() ataması yapıldı.

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10) #1.3.5. ve 7. indisler silindi.

#silme işlemini kalıcı hale getirmek için atama yapmamız gerekir. atama yapmadan silm işlemi yapmak istiyorsak inplace argümanını kullanırız.
#df.drop(delete_index, axis=0, inplace=True) şeklinde kullanılır.

###
#Değişkenleri İndexe Çevirmek
###

df["age"].head()
df.age.head()

df.index = df["age"]
df.drop("age", axis=1).head()#silme işlemi sütundan yapılacğı için axis 1 olmalıdır.
df.drop("age", axis=1, inplace=True)

df.index

df["age"] = df.index
df.head()
df.drop("age", axis=1, inplace=True)
df.head()

df = df.reset_index() #indexte yer alan değeri sütun kısmına ekliyen method.
df.head()

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None) #veri setindeki noktalı kısımlardan kurtulmamızı sağlar
df = sns.load_dataset("titanic")
df.head()

"age" in df #değişken veri setinde var mı sorgusu için kullanılır.
df.age.head()

df["age"].head()
type(df["age"].head()) #buradaki tip bilgisi pandas seriesdir. Eğer dataframe olarak kalmasını istiyorsak "age" iki köşeli parantez içinde olmalıdır.
type(df[["age"]].head())

df[["age","alive"]]

col_names = ["age", "adult_male", "alive"] #liste oluşturduk ve aramyı liste içeriğinden yapacağız
df[col_names]
df["age2"] = df["age"]**2
df.head()
df["age3"] = df["age"]/ df["age2"]

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, df.columns.str.contains("age")].head() #label based(loc)= seçim yapamk için, := tüm satırları seç, contains= stringlere uygulanan ve içeriğine string isteyen bir method
df.loc[:, ~df.columns.str.contains("age")].head() #~ = işareti değildir anlamında kullanılır.


###
#iloc and loc(integer based selection, label based selection)
###

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()


df.iloc[0:3] #integer based 3'e kadar 3 dahil değil
df.iloc[0, 0]

df.loc[0:3] #label based isimlendirmenin kendisini seçiyor.

df.iloc[0:3, 0:3]# her iki indexinde integerla belirtilmes gerekiyor.
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]


###
#Koşullu işlemler
###

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count() #yaşı 50den büyük olanların sayısı

df.loc[df["age"] > 50, "class"].head() #yaşı 50den büyük olanların sınıf bilgisi
df.loc[df["age"] > 50, ["age", "class"]].head() #koşul ve iki stün seçtik
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head() #yaşı 50den büyük ve erkek olan
df_new = df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg")
       | (df["embark_town"] == "Southampton")),
       ["age","class", "embark_town"]]

df_new["embark_town"].value_counts()

###
#Toplulaştırma ve Gruplama (Aggregation & Grouping)
###

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()


df["age"].mean() #yaş ortalaması alındı
df.groupby("sex")["age"].mean() # Cinsiyetlere göre yaş ortalaması
df.groupby("sex").agg({"age" : "mean"}) #Bu kullanım üstteki kullanıma göre daha faydalıdır ve birden fazla agg işlemi yapılabilir.
df.groupby("sex").agg({"age" : ["mean", "sum"]}) #mean ve sum işlemi birlikte yapılmıştır.
df.groupby("sex").agg({"age" : ["mean", "sum"],
                       "embark_town" : ["count"]}) #Bu noktada pivottable() fonksiyonuna ihtiyacımız var.
df.groupby("sex").agg({"age" : ["mean", "sum"],
                       "survived" : "mean"})
df.groupby(["sex", "embark_town"]).agg({"age" : ["mean", "sum"],
                                        "survived" : "mean"}) #kırılma işlemini(agg) cinsiyet + embark_tona göre yaptık.
df.groupby(["sex", "embark_town", "class"]).agg({"age" : ["mean", "sum"],
                                        "survived" : "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
       "age" : ["mean", "sum"],
       "survived" : "mean",
       "sex" : "count"})

###
#Pivot Table
###

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embark_town") #Satırlarda cinsiyet bilgisi, stünlarda embark bilgisi, kesişim kısmında ise survived değişkeninin ortalaması var
#pivot_table önem tanım gereği valueların ortalamsını alır
df.pivot_table("survived", "sex", "embark_town", aggfunc="std") #kesişim kısmının standart sapması hesaplanmış oldu
df.pivot_table("survived", "sex", ["embarked", "class"]) #stünlara bir kırılım daha eklendi
#sayısal bir değişkeni eklemek istersek değişkeni kategorik değişkene çevirmemiz gerekiyor
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90]) #cut fonksiyonuyla yaşları belli aralıklara bölme işlemini yaptık

df.pivot_table("survived", "sex", "new_age")# yaş kategorik değişkeni kırılımında ve cinsiyet kırılımında hayatta kalma oranalrını hesapladık.

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option('display.width', 500)#kod çıktısında / ları engellemek tüm kodu yanyana yazdırmak için kullanılır

###
#Apply ve Lambda
###
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
df["age2"]/10
df["age3"]/10

for col in df.columns:
       if "age" in col:
              df[col] = df[col]/10


df[["age","age2","age3"]].apply(lambda x: x/10).head() #yukarıdaki işlemi apply ve lambda kullanarak yazdık
df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head() #yaş seçim işini daha sesitematik hale getirdik

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head() # yaş - yaşın ortalaması / standart sapma

def standart_scaler(col_name):
       return(col_name - col_name.mean())/col_name.std()
df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

###
#Concat il Birleştirme işlemleri
###
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5,3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"]) #1. argüman(m) rastgele bir veri yapısı 2.argümana değişkenlerin isimlerini ver
df2 = df1 + 99

pd.concat([df1, df2]) #dataframeleri concat fonksiyonuyla birleştirdik
#ancak indeks 0, 1, 2, 3, 4 0, 1, 2, 3, 4 şeklinde ilerlemiş bunu düzeltmek için
pd.concat([df1, df2], ignore_index=True) #iki dateframein indeks bilgisini tuttuğu için böyle bir hata aldık indeksleri sıfırladık ve sorunu çözdük
#concatı incelemk için ctrl sol tık

###
#Merge ile Birleştirm işlemi
###


df1 = pd.DataFrame({'employees': ['John', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'dennis', 'mark', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]}) #veri yapıları dışardan içeriye = sözlük, string, liste, integer

pd.merge(df1, df2, on='employees') #employees e göre birleştirdi

#Amaç: her çalışanın müdür bilgisine erişmek istiyoruz
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'mustafa', 'erva']})

pd.merge(df3, df4, on='group')