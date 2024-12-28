#1. Exploratory Diabetes Analysis
#2. Data Preprocessing & Feature Engineering
#3. Base Models
#4. Automated Hyperparameter Optimization
#5. Stacking & Ensemble Learning
#6. Prediction for a New Observation
#7. Pipeline Main Function
from audioop import cross

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from yellowbrick.contrib.wrapper import classifier

from LojistikRegresyon import cv_results


#####1. Exploratory Diabetes Analysis

def check_df(dataframe, head=5):
    print("#####################")
    print(dataframe.shape)
    print("#####################")
    print(dataframe.dtypes)
    print("#####################")
    print(dataframe.head(head))
    print("#####################")
    print(dataframe.tail(head))
    print("#####################")
    print(dataframe.isnull().sum())
    print("#####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("#####################")


def cat_summary (dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name] .value_counts() / len(dataframe)}))
    print("####################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
    plt.show(block=True)


def num_summary (dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print (dataframe [numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
    plt.title(numerical_col)
    plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot = True, linevidths = 0.5, annot_kws = {'size': 12}, linecolor = 'w', cmap = 'RdBu')
    plt.show(block=True)


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

df = pd.read_csv("datasets/diabetes.csv")
check_df(df)

#Değişken tğrlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th = 5, car_th = 20)


for col in cat_cols:
    cat_summary(df, col)

df[num_cols].describe().T

#for col in num_cols:
#   num_summary(df, col, plot=True)

#Sayısal değişkenlerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

#2. Data Preprocessing & Feature Engineering

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if (dataframe[col_name] > up_limit).any() or (dataframe[col_name] < low_limit).any():
        return True
    else:
        return False


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df.head()

df.columns = [col.upper()for col in df.columns]


#Glucose
df['NEW_GLUCOSE_CAT'] = pd.cut(x=df['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetese"])

#Age
df.loc[(df['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

#BMI
df['NEW_BMI_RANGE'] = pd.cut(x=df['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                             labels=["underweight", "healty","overweight", "obese"])

#BloodPressure
df['NEW_BLOODPRESSURE'] = pd.cut(x=df['BLOODPRESSURE'], bins=[-1, 79, 89, 123], labels=["normal", "hs1", "hs2"])

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th = 5, car_th = 20)

for col in cat_cols:
    cat_summary(df, col)

for col in cat_cols:
    target_summary_with_cat(df, "OUTCOME", col)

cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

df = one_hot_encoder(df, cat_cols, drop_first=True)
check_df(df)

#Son güncel değişken türleri
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th = 5, car_th=20)

replace_with_thresholds(df, "INSULIN")

#Standartlaştırma
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

check_df(X)

#Daha sonrasında yapılan işlemleri fonksiyonlaştırdık

#3. Base Model

def base_models(X, y, scoring="roc_aud"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                    ('KNN', KNeighborsClassifier()),
                    ("SVC", SVC()),
                    ("CART", DecisionTreeClassifier()),
                    ("RF", RandomForestClassifier()),
                    ('Adaboost', AdaBoostClassifier()),
                    ('GBM', GradientBoostingClassifier()),
                    ('XBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                    ('LightGBM', LGBMClassifier())
                    # ('CatBoost', CatBoostClassifier(verbose=False))
                    ]

for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
    print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}")


#4. Automated Hyperparameter Optimization

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ('CART', DecisionTreeClassifier(), cart_params),
               ('RF', RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print ("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"######### {name} #########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f" {scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f" {name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)