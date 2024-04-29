import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns

#Duomenu valymas ir paruosimas naudojimui
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
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

#---Analizuojame populiariausius atsiskaitymo budus---

atsiskaitymo_budas = (df['Payment Method'].value_counts()
                                       .plot(kind='pie', y='popurliariausias_atsiskaitymo_budas', autopct='%1.0f%%',
                                             colors=['pink', 'skyblue', 'gold', 'orchid']))
plt.title('Atsiskaitymo būdų analizė')
plt.show()

#Atsiskaitymo budu suskaiciavimas nuo didziausio iki maziausio
populiariausias_atsiskaitymo_budas = df['Payment Method'].value_counts()
print(populiariausias_atsiskaitymo_budas)

#---inventorizacijos kiekio plaiakymas pagal kategorija---
prekiu_kiekis_pagal_kategorija = df.groupby('Product Category')['Quantity'].sum()
print(prekiu_kiekis_pagal_kategorija)

#---kurie klientai gryzta dazniausiai---

top_10_klientai_gryzta_dazniausiai = df['Customer Name'].value_counts().head(10)
print(top_10_klientai_gryzta_dazniausiai)

#---Kiek klientu renkasi prenumerata---

klientu_prenumerata = (df['Churn'].value_counts().plot(kind='pie', y='klientu_prenumerata', autopct='%1.0f%%',
                                             colors=['pink', 'skyblue', 'gold', 'orchid']))
plt.title('Klientų prenumeratų analizė')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), labels=["0-Atsisakė", "1-Sutiko"])
plt.show()

klientu_prenumerata = df['Churn'].value_counts()
print(klientu_prenumerata)










