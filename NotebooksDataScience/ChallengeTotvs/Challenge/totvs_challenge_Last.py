import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Here i get the json archive that do reference abrout the CHALLENGETOTVS data
df = pd.read_json("E:\\Projetos\\ChallengeTotvs\\Challenge\\challenge.json") 

print("Exibindo primeiras linhas do DataSet: ")
print(df.head())

#    customer_code: unique id of a customer;
#    branch_id: the branch id where this order was made;
#    sales_channel: the sales channel this order was made;
#    seller_code: seller that made this order;
#    register_date: date of the order;
#    total_price: total price of the order (sum of all items);
#    order_id: id of this order;
#    quantity: quantity of items, given by item_code, were bought;
#    item_total_price: total price of items, i.e., quantity* price;
#    unit_price: unit price of this item;
#    group_code: which group this customer belongs;
#    segment_code: segment this client belongs;
#    is_churn: if this client is set as a churn.

#contaChurn = df.groupby(['customer_code', 'branch_id', 'sales_channel',  'seller_code', 
#                        'register_date',  'total_price', 'order_id', 'quantity', 'item_code',
#                           'item_total_price', 'unit_price', 'group_code', 'segment_code']).is_churn.count()  
#Soma_Valor = df.groupby(['Data', 'UF', 'Municipio']).Valor.sum() 
#Preciso  criar tabela paralela por cliente
#Qual cliente mais comprou
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

target = 'is_churn'

categorical_features = ['branch_id', 'customer_code', 'group_code', 'item_code',
       'order_id', 'sales_channel', 'segment_code', 'seller_code']

numerical_features = ['item_total_price', 'quantity', 'total_price', 'unit_price', 'Date_Int']



df_teste = df_clean
df_teste['Date']= pd.to_datetime(df_teste['register_date'])
#df_teste['Date'].astype('datetime64[ns]')
df_teste['Date_Int'] = df_teste['Date'].astype(np.int64)
df_teste = df_teste.drop(["register_date"],axis = 1) 
print(df_teste.info())
print(df_teste.head())

df_filter = df_teste[(df_teste.unit_price < 100.000000) ]#& (airports.type == 'seaplane_base')]
print("teste filter")
print(df_filter.info())
print(df_filter.head())
print("bloxplot filter")
boxplot_df_filter = df_filter.boxplot(column=numerical_features)
print(boxplot_df_filter)

df_teste = df_filter

print(df_teste[numerical_features].describe())

boxplot = df_teste.boxplot(column=numerical_features)
print(boxplot)

#df_teste[numerical_features].hist(bins=30, figsize=(10, 7))
#plt.show()

fig, ax = plt.subplots(1, 4, figsize=(14, 4))
df_teste[df_teste.is_churn == '0'][numerical_features].hist(bins=30, color="blue", alpha=0.5, ax=ax)
plt.show()
df_teste[df_teste.is_churn == '1'][numerical_features].hist(bins=30, color="red", alpha=0.5, ax=ax)
plt.show()



print("Sairam:             ", df_teste['is_churn'].value_counts()[1])
print("Continuam clientes: ", df_teste['is_churn'].value_counts()[0])

'''
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
X =  df_teste.drop(['is_churn'], axis=1)
y =  df_teste['is_churn']

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
X = df_teste.drop(labels = ["is_churn"], axis = 1)
y = df_teste["is_churn"].values

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
print(feat_importances.sort_values(ascending = False))


'''
df_teste.groupby(["group_code", "is_churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 


df_teste.groupby(["segment_code", "is_churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 
plt.show()
'''