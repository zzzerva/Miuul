# Random Forest: Birden fazla ağacın bir araya gelip tahmin etmesi olayıdır
# Ağaçlar için gözlemler bootstrap rastgele örnek seçim yöntemi ile değişken random subspace yöntemi ile seçilir
# Bagging: Topluluk öğrenme noktası. Birbirinden farklı ağaçların fit edilmesi
# Birçok farklı ağaç yöntemine uygulanabilir


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pprint import pprint

from CART import cart_final
from LojistikRegresyon import cv_results

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("datasets/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

rf_model = RandomForestClassifier(random_state=17)

pprint(rf_model.get_params())

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

pprint(rf_params)

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


def plot_importance(model, features, num=len(X), save=False):
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

plot_importance(rf_final, X, num=5)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Train Score", color='b')
    plt.plot(param_range, mean_test_score, label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(scoring)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")


####Gradient Boosting Machines(GBM)
#AdaBoost: Zayıf sınıflandırıcıların bir araya gelerek güçlü bir sınıflandırıcı oluşturması fikrine dayanır
#Zayıf öğreniciler bir araya gelerek güçlü bir öğrenici oluşturabilir mi??? sorundan doğmuşutur

#GBM: Hatalar üzerine tek bir tahminsel model formunda olan modeller serisi kurulur

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.75917
cv_results['test_f1'].mean()
#0.63423
cv_results['test_roc_auc'].mean()
#0.82601

gbm_params = {"learning_rate": [0.01, 0.1],
             "max_depth": [3, 8, 10],
             "n_estimators": [100, 500, 1000],
             "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.77349
cv_results['test_f1'].mean()
#0.66084
cv_results['test_roc_auc'].mean()
#0.83532


###XGBoost GBM'in hız ve tahmin performansını arttırmak üzere optimize edilmiş; ölçeklenebilir ve farklı platformlara entegre edilebilir versiyonudur

xgboost_model = XGBClassifier(random_state=17)

xgboost_model.get_params()

cv_results = cross_validate(xgboost_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.74347
cv_results['test_f1'].mean()
#0.6221
cv_results['test_roc_auc'].mean()
#0.7921

xgboost_params = {"learning_rate": [0.1, 0.01],
                "max_depth": [5, 8, None],
                "n_estimators": [100, 500, 1000],
                "colsample_bylevel": [None, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.75917
cv_results['test_f1'].mean()
#0.64555
cv_results['test_roc_auc'].mean()
#0.81417

#####LightGBM
#XGBoost'un eğitim süresi performansını arttırmaya yönelik geliştirilen bir diğer GBM türüdür
#Level-wise stratejisi yerine Leaf-wise ile daha hızlıdır.


lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.747449
cv_results['test_f1'].mean()
#0.62411
cv_results['test_roc_auc'].mean()
#0.79902

lgbm_params = {"learning_rate": [0.01, 0.1],
                "n_estimators": [100, 300, 500, 1000],
                "colsample_bylevel": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
                "n_estimators": [200, 300, 350, 400],
                "colsample_bylevel": [0.9, 0.8, 1]}

#CatBoost: kategorik değişkenler ile otomatik olarak mücadele edebilen GBM türevidir

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.77
cv_results['test_f1'].mean()
#0.65
cv_results['test_roc_auc'].mean()
#0.83

####Feature Importance

def plot_importance(model, features, num=len(X), save=False):
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


plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
