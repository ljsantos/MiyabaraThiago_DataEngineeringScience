
import pandas as pd 
  
df = pd.read_json('C:\\Users\\thiago.yoshiaki\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\pysignfe\\.vscode\\challenge\\challenge.json') 
#print(df) 

#df.info()


print(df.describe())

print(df.is_churn.value_counts())



print(df[df['is_churn'].isnull()])


print(df)




