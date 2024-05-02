import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')


# Išskiriami požymiai
selected_features = ['Product Category', 'Product Price', 'Quantity', 'Purchase Date']
df_selected = df[selected_features]

# Konvertuojama į datetime formatą
df_selected['Purchase Date'] = pd.to_datetime(df_selected['Purchase Date'])

df_selected['Year'] = df_selected['Purchase Date'].dt.year
df_selected['Month'] = df_selected['Purchase Date'].dt.month
df_selected['Day'] = df_selected['Purchase Date'].dt.day
df_selected['Hour'] = df_selected['Purchase Date'].dt.hour
# print(df_selected.head())

# Sumuojami pirkimai pagal metus ir mėnesius
df_grouped = df_selected.groupby(['Year', 'Month']).agg({'Quantity': 'sum'}).reset_index()

# Duomenys ruošiami modeliui
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_grouped['Quantity'].values.reshape(-1, 1))

#Laiko eilutės
look_back = 12  # Imame 12 mėnesių duomenis
X, Y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i, 0])
    Y.append(scaled_data[i, 0])

X, Y = np.array(X), np.array(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Padalijama į mokymo ir testavimo rinkinius
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# modelis
model = Sequential([
    GRU(units=133, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    GRU(units=133),
    Dense(1)
])
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# Treniravimas
model.fit(X_train, Y_train, epochs=33, batch_size=16, validation_data=(X_test, Y_test), verbose=1)

# Prognozė
predictions = model.predict(X_test)

# Atstatomas orginalus mastelis
predictions = scaler.inverse_transform(predictions)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(Y_test, predictions))
print("RMSE:", rmse)

# Įvertinimas naudojant R^2
r2 = r2_score(Y_test, predictions)
print("R2:", r2)

# Atvaizduoti rezultatus
plt.plot(Y_test, label='Pagal faktą')
plt.plot(predictions, label='Prognozė')
plt.xlabel('Mėnesis')
plt.ylabel('Kiekis')
plt.title('Faktiniai ir prognozuojami pirkimai')
plt.legend()
plt.show()

# Geriausi hiperparametrai: {'batch_size': 16, 'epochs': 11, 'lstm_units': 133, 'optimizer': 'rmsprop'}
# RMSE: 0.06881850797856753
# R2: -2.086435104145237

#SU LSTM ir 33 epochom
# RMSE: 385.84462729521374
# R2: 0.022753505179238354


#SU GRU ir 33 epochom
# RMSE: 380.98184157275733
# R2: 0.04723068818549825

