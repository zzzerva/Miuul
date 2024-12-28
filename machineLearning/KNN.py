#KNN: En yakın komşuya göre var olamayn verilerin tahmin edilmesi
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

####1. Exploratory Data Analysis

df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

####2. Data Preprocessing & Featue Engineering

y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)

x_scaled = StandardScaler().fit_transform(x)

x = pd.DataFrame(x_scaled, columns=x.columns)

####3. Modeling & Prediction

knn_model = KNeighborsClassifier().fit(x, y)

random_user = x.sample(1, random_state=45)

knn_model.predict(random_user)#knn modelin öğrenme yöntemine göre hangi ögeyi predict etmesini istediğimizi yazdık

####Model Evulation

#Confusion matrix için pred:
y_pred = knn_model.predict(x)

#AUC için y_prob
y_prob = knn_model.predict_proba(x)[:, 1]

print(classification_report(y, y_pred))
#acc 0.83
#f1 0.74
#AUC
roc_auc_score(y, y_prob)
#%90

#5 Katlı çapraz dorulama

cv_results = cross_validate(knn_model,
                            x,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# test_acc: 0.73
# test_f1: 0.59
# test_roc_auc: 0.78

#Örnek sayısı arttırılabilir.
#Veri ön işleme
#İlgili algoritma için optimizasyon yapılabilir

knn_model.get_params()

#Hiperparametre Optimizasyonu

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(x, y)
#verbose: rapor bekler misisn, n_jobs = -1 işlemcinin tamaını kullan demek


knn_gs_best.best_params_

#### 6.Final Model

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(x, y)
cv_results = cross_validate(knn_final,
                            x,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#final_acc: 0.76
#final_f1: 0.61
#final_roc_auc: 0.81