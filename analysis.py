import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns

#Duomenu valymas ir paruosimas naudojimui
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')
pd.set_option('display.max_columns', None)
# print(df.describe())

#Tikriname eiluciu skaiciu
print(len(df))

# -Tikriname NaN reiksmes
# print(df.isnull().sum())

# -sutvarkome returns skilti, pasauliname NaN reiksmes
df['Returns'] = df['Returns'].fillna(0).astype(int)

# -convertuojame pirkimo data to datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
df['Purchase Year'] = df['Purchase Date'].dt.year
df['Purchase Month'] = df['Purchase Date'].dt.month_name()

df[['Purchase Date','Purchase Year','Purchase Month']]


#Analizuojame duomenis

# -Analizuojame duomenis pagal lyti(skaiciuojame pirkimu vidurki, )

pirkimai_pagal_lyti = df.groupby('Gender')['Total Purchase Amount'].mean()
# print(pirkimai_pagal_lyti)

mokejimo_metodas_pagal_lyti = df.groupby('Gender')['Payment Method'].value_counts()
# print(mokejimo_metodas_pagal_lyti)

sns.countplot(data=df, order=mokejimo_metodas_pagal_lyti.index,)
plt.legend()
plt.show()
