import os
import io
import platform
import itertools

import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt

import _pickle as pkl

import seaborn as sns
import warnings

from pylab import rcParams

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


warnings.filterwarnings('ignore')

print('python version', platform.python_version())
print('pandas version', pd.__version__)
print('numpy version', pd.__version__)
#print('sklearn version', sklearn.__version__)
#print('matplotlib version', matplotlib.__version__)
print('seaborn version', sns.__version__)


df = pd.read_json("E:\\Projetos\\ChallengeTotvs\\Challenge\\challenge.json") 

#df.profile_report() 
#profile = df.profile_report(title='PandasProfilingReport')
#profile.to_file(outputfile="./ChaTOTVS.html")

print("Exibindo primeiras linhas do DataSet: ")
print(df.head())

print("Describe da tabela: ")
print(df.describe())

print("Verificando a quantidade de valores de cada coluna: ")
print(df.count())


print("\Contagem de numeros nulos: \n", df.isnull().sum())



# remove todas as linhas cujo scouts são NANs 
df_clean = df.dropna()
print('qtde. de clientes com dados, sem valores nulos: ', df_clean.shape[0])


print("\Contagem de numeros nulos: \n", df_clean.isnull().sum())



print("Verificando a quantidade de valores de cada coluna na tabela limpa: ")
print(df_clean.count())


print("Exibindo nome das colunas antes: ")
print(df_clean.columns)

'''
df_clean.columns = ['branch_id', 'customer_code', 'group_code', 'is_churn', 'item_code',
       'item_total_price', 'order_id', 'quantity', 'register_date',
       'sales_channel', 'segment_code', 'seller_code', 'total_price',
       'unit_price']
'''

#df_clean = df_clean.drop(["branch_id", "register_date"],axis = 1)

#df_clean = df_clean.drop(["branch_id", "register_date"],axis = 1)

print("Exibindo nome das colunas atuais: ")
print(df_clean.columns)

print("\nUnique Values: \n", df_clean.nunique())

print('Data Exploration: ')
print(df_clean.info())

print("Exibindo primeiras linhas do DataSet: ")
print(df_clean.head())


print("Sairam:             ", df_clean['is_churn'].value_counts()[1])
print("Continuam clientes: ", df_clean['is_churn'].value_counts()[0])

dic = { 'item_code': 'last',
       'item_total_price': 'mean',
       'order_id': 'first',
       'quantity': 'mean',
       'seller_code': 'first',
       'total_price': 'mean',
       'unit_price': 'mean',
      }

df_ordem = df_clean[df_clean.columns]\
    .sort_values(['customer_code', 'order_id'])\
    .groupby(['customer_code', 'order_id', 'register_date' ])

print(df_ordem.head())

'''
y = df_clean["is_churn"].value_counts()
#print (y)
sns.barplot(y.index, y.values)
plt.show()

y_True = df_clean["is_churn"][df_clean["is_churn"] == 1]
print ("Churn Percentage = "+str( (y_True.shape[0] / df_clean["is_churn"].shape[0]) * 100 ))
'''

'''
def bar_chart(feature):
    sairam = df_clean[df_clean['is_churn']==1][feature].value_counts()
    naosairam = df_clean[df_clean['is_churn']==0][feature].value_counts()
    df_churn = pd.DataFrame([sairam,naosairam])
    df_churn.index = ['Sairam','Nao Sairam']
    df_churn.plot(kind='bar',stacked=True, figsize=(10,5))

bar_chart('group_code')
plt.show()

bar_chart('group_code')
plt.show()
'''

'''
Machine Learning Algorithm Training
'''

'''
Random Forest
'''
X =  df_clean.drop(['is_churn'], axis=1)
y =  df_clean['is_churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, random_state=0)  
classifier.fit(X_train, y_train)  
predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions ))  
print(accuracy_score(y_test, predictions ))

'''
Logistic Regression
'''
X = df_clean.drop(labels = ["is_churn"],axis = 1)
y = df_clean["is_churn"].values

# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)

from sklearn import metrics
prediction_test = model.predict(X_test)

# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))

# To get the weights of all the variables
weights = pd.Series(model.coef_[0],
 index=X.columns.values)
print(weights.sort_values(ascending = False))

feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')

df_clean.groupby(["group_code", "is_churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 
plt.show()

df_clean.groupby(["segment_code", "is_churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 
plt.show()
