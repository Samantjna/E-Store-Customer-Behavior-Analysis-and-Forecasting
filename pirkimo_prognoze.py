import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')

#Duomenu valymas ir paruosimas naudojimui
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv', nrows=1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# print(df.describe())

# konvertuojame datos laika i laiko zyma sekundemis nuo epochos
df['Purchase Date'] = df['Purchase Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())

X = df[['Purchase Date', 'Product Category', 'Product Price']]

y = df['Quantity']

# kategoriniai
categorical_features = ['Product Category']

# transformuojami duomenys
trans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')

# Transformuojami kategoriniai nudojant OneHotEncoder
X_encoded = trans.fit_transform(X)

# Padalinami duomenys i mokymo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# apmokomas modelis
model = LinearRegression()
model.fit(X_train, y_train)

# prognozavimas
y_pred = model.predict(X_test)

# modelio tikslumas
mse = mean_squared_error(y_test, y_pred)
print("Vidutinis kvadratinis nuokrypis:", mse)


# sukuriamas df
df_results = pd.DataFrame({'Faktinė vertė': y_test, 'Prognozuota vertė': y_pred})

#atvaizduojama
plot = sns.lmplot(x='Faktinė vertė', y='Prognozuota vertė', data=df_results)
plot.set_titles('Tiesinė regresija: Faktinės vs. Prognozuotos vertės')
plt.show()