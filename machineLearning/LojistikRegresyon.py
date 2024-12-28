####
#Diabetes Prediction with Logitic Regresion
####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.interchange import dataframe

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score,confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score



def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit



def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if (dataframe[col_name] > up_limit).any() or (dataframe[col_name] < low_limit).any():
        return True
    else:
        return False



def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape

#Target'ın Analizi

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

df.describe().T

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col] #Bağımlı değişkeni çıkardık

for col in cols:
    plot_numerical_col(df, col)


df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)

df.isnull().sum()
df.describe().T #eksik değer yok gibi yakalaşılacak ancak nan ler 0 ile değiştirilmiş

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")


for col in cols: #Aykırı değerlerden etkilenmiyorfor col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


########
#Model & Prediction
########

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

##Model Evaluation

def plot_confusion_matrix(x, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

#Accuracy: 0.78
#Precision: 0.74
#Recall: 0.58
#F1 Score: 0.65


#ROC AUC

y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_pred)
#0.737
#Modeli, modelin öğrenildiği veri üzerinde test ettik.
#Dolayısıyla aldığımız sonuçların test edilmesi gerekiyor

##Holdout

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)

y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

#Eski
#Accuracy: 0.78
#Precision: 0.74
#Recall: 0.58
#F1 Score: 0.65

###Yeni
#Accuracy: 0.77
#Precision: 0.79
#Recall: 0.53
#F1 Score: 0.63

#Öncekiyle farklı sonuçlar yok ancak yinede kotrol edilmesi gerekiyor.Model görmediği veriler üzerinde başarısız olabilir(?)

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


roc_auc_score(y_test, y_prob)

####10 Katlı çapraz doğrulama

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_val_score(log_model,
                X, y,
                cv=5,
                scoring=('accuracy','precision','recall','f1','roc_auc'))

cv_results['test_accuracy'].mean()

X.columns
random_user = X.sample(1, random_state=45)
log_model.predict(random_user)