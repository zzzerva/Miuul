import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

####
#Simple Linear Regression with Gradient Descent from Scratch
####

#Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0 #hata kareler toplamını 0 a eşitledik

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


#update weight
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)

    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i] #1. iterasynun gözlem değeri
        y = Y[i] #1. iterasyonun gerçek değeri
        b_deriv_sum += (y_hat - y) #gözlem değeri - gerçek değer (sabitin kısmi türevi)!!!!
        w_deriv_sum += (y_hat - y) * X[i] #(ağırlığın kısmi türevi)!!!!!
    #gradian distance??
    nem_b = b - (learning_rate * 1 / m * b_deriv_sum) #eski değerleri güncelliyoruz
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return nem_b, new_w


def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                     cost_function(Y, initial_b, initial_w, X)))
    # Güncelleme olmadn önceki hatayı yazdırdık
    b = initial_b
    w = initial_w
    cost_hitory = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate) #1. iterasyondan sonraki hali yeniden günceledik
        mse = cost_function(Y, b, w, X) #hatayı tekrar hesapladık
        cost_hitory.append(mse)


        if i % 100 == 0:
            print("iter={:d}  b={:.2f}  w={:.4f}  mse={:.4}".format(i, b, w, mse))
        #raporlama yapılmaıs için

    print("After {0} iteration b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_hitory, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

#Parametre: veri setinden bulunurken hiperparametre veri setinden bulunamaz ve kullanıcı tarafından bulunması gerekir


#hiperparametre
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

train(Y, initial_b, initial_w, X, learning_rate, num_iters)