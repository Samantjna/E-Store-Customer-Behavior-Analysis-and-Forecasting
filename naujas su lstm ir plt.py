import pandas as pd
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

# Įkeliami duomenis
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')



# Duomenų paruošimas
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
df['Purchase Date'] = df['Purchase Date'].apply(lambda x: x.timestamp())
X = df[['Purchase Date', 'Product Category', 'Product Price']]
y = df['Quantity']


# Kategoriniai
categorical_features = ['Product Category']


# Transformuojami kategoriniai
trans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_encoded = trans.fit_transform(X)


# Duomenų standartizavimas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


# Padalinami duomenys į mokymo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)


# Pertvarkomi duomenys į trimačius masyvus
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# Modelis
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(1)
])


# Kompiliavimas
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mse'])


# Ankstyvasis sustabdymas
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Apmokymas kviečiant ankstyvajį sustabdymą
history = model.fit(X_train_reshaped, y_train, epochs=32, batch_size=49, validation_split=0.2, callbacks=[early_stopping])


# Prognozės
y_pred = model.predict(X_test_reshaped)


# Vertinimai
loss, mse = model.evaluate(X_test_reshaped, y_test)
print("MSE:", mse)
r2 = r2_score(y_test, y_pred)
print("R2:", r2)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)


# Laiko serija iki 2025
future_dates = pd.date_range(start=pd.to_datetime(df['Purchase Date'], unit='s').max(), end='2025-12-31', freq='MS')


# Papildomos datos, kad būtų vienodi ilgiai
additional_dates = pd.date_range(start=future_dates[-1] + pd.Timedelta(days=1), periods=1000-len(future_dates), freq='MS')


# Prailginama future_dates
extended_future_dates = pd.concat([pd.Series(future_dates), pd.Series(additional_dates)])


# Paruošimas prognozavimui
future_data = pd.DataFrame({
    'Purchase Date': extended_future_dates.astype(np.int64) // 10**9,  # Paverčiame laiką į sekundes
    'Product Category': np.repeat(df['Product Category'].iloc[0], len(extended_future_dates)),
    'Product Price': np.repeat(df['Product Price'].iloc[0], len(extended_future_dates))
})


# Transformuojami ir standartizuojami duomenys
future_X = scaler.transform(trans.transform(future_data))


# Prognozės laikams
future_pred = model.predict(future_X.reshape(future_X.shape[0], 1, future_X.shape[1]))


# DF su faktiniais duomenimis ir prognozėmis
results_df = pd.DataFrame({
    'Date': np.concatenate((pd.to_datetime(df['Purchase Date'], unit='s')[y_test.index], extended_future_dates)),
    'Actual': np.concatenate((y_test, [None]*len(extended_future_dates))),
    'Predicted': np.concatenate((y_pred.flatten(), future_pred.flatten()))
})


# Konvertuojama data į tekstą
results_df['Date'] = pd.to_datetime(results_df['Date']).dt.strftime('%Y-%m')


# Matplotlib
plt.figure(figsize=(15, 6))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual')
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted')
plt.title('Faktinės ir prognozuotos pirkimų reikšmės')
plt.xlabel('Data')
plt.ylabel('Kiekis')
plt.legend()
plt.xticks(rotation=90)
plt.xlim('2023-06', '2024-06') # Pasirenkame datos intervalą
plt.tight_layout()
plt.show()