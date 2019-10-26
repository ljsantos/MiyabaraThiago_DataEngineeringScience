import os
import io
import platform
import itertools

import numpy as np
import pandas as pd

df = pd.read_json('E:\Projetos\\ChallengeTotvs\\Challenge\\challenge.json') 

print("Exibindo primeiras linhas do DataSet: ")
print(df.head(30))

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

df_clean.columns = ['branch_id', 'customer_code', 'group_code', 'saiu', 'item_code',
       'item_total_price', 'order_id', 'quantity', 'register_date',
       'sales_channel', 'segment_code', 'seller_code', 'total_price',
       'unit_price']

#df_clean = df_clean.drop(["branch_id", "register_date"],axis = 1)

print("\nUnique Values: \n", df_clean.nunique())

print('Data Exploration: ')
print(df_clean.info())

print("Exibindo primeiras linhas do DataSet: ")
print(df_clean.head())


print("Sairam:             ", df_clean['saiu'].value_counts()[1]);
print("Continuam clientes: ", df_clean['saiu'].value_counts()[0])

X =  df_clean.drop(['saiu'], axis=1)
y =  df_clean['saiu']

