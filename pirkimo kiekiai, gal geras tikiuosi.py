import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Įkeliam duomenis
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv', nrows=50000)

# Duomenų paruošimas
df['Purchase Date'] = df['Purchase Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())
X = df[['Purchase Date', 'Product Category', 'Product Price']]
y = df['Quantity']

# Kategoriniai kintamieji
categorical_features = ['Product Category']

# Transformuojami kategoriniai kintamieji
trans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_encoded = trans.fit_transform(X)

# Skalavimas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Padalinami duomenys į mokymo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelis
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

# Kompiliavimas
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])

# Apmokymas
history = model.fit(X_train, y_train, epochs=32, batch_size=49, validation_split=0.2)

# Prognozės
y_pred = model.predict(X_test)

# Vertinimai

loss, mse = model.evaluate(X_test, y_test)
print("MSE:", mse)


r2 = r2_score(y_test, y_pred)
print("R2:", r2)

# MAE
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)



# # Tikrosios ir prognozuojamos reikšmės
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(y_test)), y_test, color='blue', label='Tikrosios reikšmės')
# plt.scatter(range(len(y_pred)), y_pred, color='red', label='Prognozuojamos reikšmės')
# plt.title('Tikrosios ir prognozuojamos reikšmės')
# plt.xlabel('Eilės numeris')
# plt.ylabel('Kiekis')
# plt.legend()
# plt.show()