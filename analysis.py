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

# mokejimo_metodas_pagal_lyti = df.groupby('Gender')['Payment Method'].value_counts()
# print(mokejimo_metodas_pagal_lyti)

# sns.countplot(data=df, order=mokejimo_metodas_pagal_lyti.index)
# plt.legend()
# plt.show()

#hue - atskiria papildomus duomenu aspektus
sns.countplot(data=df, x='Payment Method', hue='Gender', order=df['Payment Method'].value_counts().index, palette='husl')
plt.legend(title='Lytis')
plt.show()


#klientu amziaus kategorijos
jauniausias_klientas = df['Age'].min()
print(f'Jauniausias klientas: {jauniausias_klientas}')

amzius_iki_25 = df[df['Age'] < 25].shape[0]
print(f'Klientai jaunesni nei 25 m.: {amzius_iki_25}')

klientai_nuo_25_iki_50 = df[(df['Age'] > 25) & (df['Age'] < 50)].shape[0]
print(f'Klientai nuo 25 m. iki 50 m.: {klientai_nuo_25_iki_50}')

vyriausias_klientas = df['Age'].max()
print(f'Vyriausias klientas: {vyriausias_klientas}')

amzius_nuo_50 = df[(df['Age'] > 50)].shape[0]
print(f'Klientai nuo 50 m.: {amzius_nuo_50}')

age_groups = ['Customers under 25 years of age', 'Customers aged 25 to 50', 'Customers aged 50+']
quantity = [amzius_iki_25, klientai_nuo_25_iki_50, amzius_nuo_50]

df = pd.DataFrame({'Age groups': age_groups, 'Quantity': quantity})

plt.figure(figsize=(10, 8))
sns.barplot(data=df, x='Age groups', y='Quantity', hue='Age groups', palette='husl')
plt.xlabel('Age groups', fontsize=13)
plt.ylabel('Quantity of customers', fontsize=13)
# plt.xticks(rotation=7)
plt.title('Number of customers by age group', fontsize=15)
plt.show()