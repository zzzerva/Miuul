import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


###Simple Linear Regression with DLS Using Scikit-Learn


df = pd.read_csv("datasets/advertising.csv")
df.shape
x = df[["TV"]]
y = df[["sales"]]

####MODEL

reg_model = LinearRegression().fit(x, y)
#y_hat = b + w*x

#sabit - bias
reg_model.intercept_[0] # 0 yazdaman dönerse array şeklinde geliyor

#tv'nin katsayısı - w1
reg_model.coef_[0][0] #(coefficients)

#Tahmin işlemi

#150 birimlik tv harcaması olursa ne kdar satış olması beklenir.

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T

g = sns.regplot(x=x, y=y, scatter_kws={'color': 'b', 's':9},
                ci=False, color="r") # renk ekledim. ci güven aralığı demek güven aralığını false yaptım
#Modele title eklerken dinamik eklenmiştir
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")#eksenlere yazılacak yazılar
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)#Gösterim aralığı
plt.ylim(bottom=0)
plt.show()

###Tahmin Başarısı

#MSE
y_pred =  reg_model.predict(x)
mean_squared_error(y, y_pred)
y.mean()
y.std() #bakılan bu verilere göre hata başarımız düşük

#RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.24
#MAE
mean_absolute_error(y, y_pred)
#2.54
#R-KARE
reg_model.score(x, y) #bağımsız değişkenin bağımlı değişkeni açıklama değişkeni (mesela %60 şeklinde açıklar)
#0.61

###Multiple Linear Regression

df = pd.read_csv("datasets/advertising.csv")

x = df.drop('sales', axis=1)
y = df['sales']

#Model

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)#random_state 1: oluşturduğum train testle sizin oluşturduğunuz train test aynıysa 1 yaz

x_train.shape
y_train.shape
x_test.shape
y_test.shape

reg_model = LinearRegression().fit(x_train, y_train)

#bias
reg_model.intercept_

#coefficients
reg_model.coef_

#Tahmin

#TV : 30
#radio : 10
#newspaper : 40

#2.90
#0.0468431 , 0.17854434, 0.00258619

Sales = 2.90 + 30 * 0.04 +10 * 0.17 + 40 * 0.002

yeni_veri = [[30], [10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T
reg_model.predict(yeni_veri)

#Tahmin Başarısı

#Train RMSE
y_pred = reg_model.predict(x_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#1.73 hata
#Train RKARE
reg_model.score(x_train, y_train)

#Test RMSE
y_pred = reg_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#1.41 hata

#Test RKARE
reg_model.score(x_test, y_test)

#10 Katlı CV(cross validation) RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 x,
                                 y,
                                 cv=10,
                                 scoring='neg_mean_squared_error')))
#1.69

