import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from datetime import datetime


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

# LSTM modelis
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Treniravimas
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

# Prognozė
predictions = model.predict(X_test)

# Atstatomas orginalus mastelis
predictions = scaler.inverse_transform(predictions)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Atvaizduoti rezultatus
plt.plot(Y_test, label='Pagal faktą')
plt.plot(predictions, label='Prognozė')
plt.xlabel('Mėnesis')
plt.ylabel('Kiekis')
plt.title('Faktiniai ir prognozuojami pirkimai')
plt.legend()
plt.show()


