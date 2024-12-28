#1. Exploratory Data Analysis
#2. Data Preprocessing & Feature Engineering
#3. Modeling Using CART
#4. Hyperparameter Optimization with GridSearchCV
#5. Final Model
#6. Feature Model
#7. Analyzing Model Complexity with Learning Curves(BONUS)
#8. Visualizing the Desicion Tree
#9. Extracting Desicion Rules
#10. Extrating Python/SQL/Excel Codes of Desicion Rules
#11. Prediction using Python Codes
#12. Saving and Loading Model

from distutils.command.install import install

import astor
import conda
#import graphviz
import joblib
import pydotplus

#pip install pydotplus
#pip install skompiler
#pip install astor
#pip install joblib


import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.core.common import random_state
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve

from BasitDoğrusalRegresyonModeli import x_train, x_test
from LojistikRegresyon import cv_results

#from skompiler import skompile


pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

#### 1.Exploratory Data Analysis


#### 2.Data Preprocessing & Feature Engineering

df = pd.read_csv("datasets/diabetes.csv")

y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(x, y)

#Confusion matrix için y_pred:
y_pred = cart_model.predict(x)

#AUC için y_prob:
y_prob = cart_model.predict_proba(x)[:, 1]

#Confusion matrix
print(classification_report(y, y_pred))

#AUC
roc_auc_score(y, y_prob)
#Başarım 1 çıktı. Emin olamıyoruz model aşırı öğrenmeye i düştü bunu anlamak için holdout yöntemini kullanabiliriz

####Holdout ile Başarı Değerlendirme

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(x_train, y_train)

#Train Hatası

y_pred = cart_model.predict(x_train)
y_prob = cart_model.predict_proba(x_train)[:, 1]
print(classification_report(y_train, y_prob))
roc_auc_score(y_train, y_prob)
#Train setiyle kurulan modelin başarısı hala 1

#Test hatası
y_pred = cart_model.predict(x_test)
y_prob = cart_model.predict_proba(x_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
#Model veriyi ezberlediği için(overfit) görmediği veriyi tahmin ederken hata veriyor
#random_statei 17den 45e çıkarılırsa sonuç iyice kötüleşiyor

###CV ile Başarı Değerlendirme

cart_model = DecisionTreeClassifier(random_state=17).fit(x, y)
#fit(x, y) yazılması zorunlu değildir CV direkt kendisi fit etme işlemini gerçekleştirebilir.

cv_results = cross_validate(cart_model,
                            x, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


####4. Hyperparameter Optimization with GridSearchCV

cart_model.get_params()
#min_samples_split, max_depth bunlar overfittinge sebep olabilir

cart_params = {'max_depth': range(1, 11),
               'min_samples_leaf': range(2, 20)}

cart_best_grid = GridSearchCV(cart_model, cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(x, y)

cart_best_grid.best_params_
cart_best_grid.best_score_

random = x.sample(1, random_state=45)

cart_best_grid.predict(random)

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(x, y)
cart_final.get_params()

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(x, y)

cv_results = cross_validate(cart_final,
                            x, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

####6. Feature Importance

cart_final.feature_importances_

def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=[10, 10])
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save: #save=True olursa görseli kaydeder
        plt.savefig('importance.png')

plot_importance(cart_final, x, num=5)


####7. Analyzing Model Complexity with Learning Curves (BONUS)


train_score, test_score =validation_curve(cart_final, x, y,
                                          param_name="max_depth",
                                          param_range=range(1, 11),
                                          scoring="roc_auc",
                                          cv=10)

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)
#1 parametre için 10 tane var 10 parametre için 100 tane var o yüzzden 10 tane sayı görüyoruz


plt.plot(range(1, 11), mean_train_score,
         label="Train Score", color='b')

plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("number of Max_Depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

def val_curve_params(model, x, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score =validation_curve(
        model, x=x, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Train Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

val_curve_params(cart_final, x, y, "max_depth",range(1, 11))



##import graphviz


def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_pdf(file_name)

tree_graph(model=cart_final, col_names=x.columns, file_name="cart_final.png")


tree_rules = export_text(cart_final, feature_names=list(x.columns))
print(tree_rules)


####Extracting Python Codes of Decision Rules

print(skompile(cart_final.predict).to('python/code'))
##sklearn 0.23.1 versiyonu ile yapılabiliyor

print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))


#####Saving and Loading Model

joblib.dump(cart_final, 'cart_final.pkl')

cart_model_from_disc = joblib.load("cart_final.pkl")

x = [12, 13, 20, 23, 4, 55, 12, 7]

cart_model_from_disc.predict(pd.DataFrame(x).T)