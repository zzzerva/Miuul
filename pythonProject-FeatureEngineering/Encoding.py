#####
#Encoding
#####
#Kategorik değişkendeki stringleri numerik değişkenlerer dönüştürüyoruz ki kullanılacak yöntemlerin anlaşılacağı şekile çeviriyoruz

#Label Encoding & Binary Encoding
#iki sınıf varsa(0 ve 1lerden oluşyorsa) Binary Encoding
#ikiden fazla sınıfsa Label Encoding
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from matplotlib import pyplot as plt
from Outliner import cat_cols, num_cols, cat_but_car

df = sns.load_dataset("titanic")
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
df.head()
df["sex"].head()#Binary Encode etmek istiyoruz

def load():
    data = pd.read_csv('datasets/titanic.csv')
    return data

le = LabelEncoder()
le.fit_transform(df["sex"])[0:5]
le.inverse_transform([0, 1])#0=female, 1= male

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = sns.load_dataset("titanic")

#iki sınıflı kategorik değişkenleri bulabilirsem binary encoding işlemini yapabilirim

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
#değişkenin tipi in veya float değilse eşsiz sınıf sayısı 2 ye eşitse


for col in binary_cols:
    label_encoder(df, col)
df.head()

def load_application_train():
    data = pd.read_csv('datasets/application_train.csv')
    return data

df = load_application_train()
df.head()


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
             and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col) #label encoderdan geçirdikten sonra eksik değerleri de dolduruyor
    #Eksik değerlerin doldurulması yaygın olarak tercih edilen bir seçenektir


df = sns.load_dataset("titanic")
df["embarked"].value_counts()
df["embarked"].nunique()
len(df["embarked"].unique()) #Eksik değeeleri göz önünde bulundurmak için kullanılabilir


####
#One-hot Encoding
####
#Sııflar arası fark olmadığı halde sıralamayı fark varmış gibi yaptığı için label encoding de bazen hatalar oalbiliyor

#Bir gözlem birimindeysek o gözlem birimini 1, diğerlerine 0 olacak şekilde one-hot encoding işlemi gerçekleştirilmiş olur
#dummy değişken tuzağı(kukla değişken tuzağı):Eğer değişkenler birbiri üzerinden oluşturulacak olursa bir ölçme problem ortaya çıkmaktadır
#dummy tuzağından kurtulmak için ilk sınıfı drop etmemiz gerekir

df = sns.load_dataset("titanic")
df.head()
df["embarked"].value_counts()

pd.get_dummies(df, columns =["embarked"]).head()
#Alfabetik sıraya göre embarked değerinin ilk değişkeni drop edilir
pd.get_dummies(df, columns=["embarked"], drop_first=True).head()
#Eksik değerler için bir sınıf oluşturur.
pd.get_dummies(df, columns=["embarked"], dummy_na=True).head()

pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

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

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()

####
#Rare Encoding
####
#Sınıfların gözlenme frekansına göre encode işlemi yapılır
#Kategorik değişkenelerin azlik çokluk durumu analizi
#Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analizi

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary (dataframe, col_name, plot=False):
    print (pd. DataFrame({col_name: dataframe[col_name].value_counts(),
                         "Ratio": 100 * dataframe[col_name].value_counts / len(dataframe)}))

    print("##############################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt. show()

for col in cat_cols:
    cat_summary(df, col)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATİO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [ col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                     and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)#%01lik kısmından altında kalan değerleri birleştirecek
rare_analyser(new_df, "TARGET", cat_cols)