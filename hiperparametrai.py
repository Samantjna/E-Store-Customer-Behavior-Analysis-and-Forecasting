import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scikeras.wrappers import KerasRegressor
from scipy.stats import randint as sp_randint


df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')
selected_features = ['Product Category', 'Product Price', 'Quantity', 'Purchase Date']
df_selected = df[selected_features]

# Konvertuojama į datetime formatą
df_selected['Purchase Date'] = pd.to_datetime(df_selected['Purchase Date'])

df_selected['Year'] = df_selected['Purchase Date'].dt.year
df_selected['Month'] = df_selected['Purchase Date'].dt.month
df_selected['Day'] = df_selected['Purchase Date'].dt.day
df_selected['Hour'] = df_selected['Purchase Date'].dt.hour

# Sumuojami pirkimai pagal metus ir mėnesius
df_grouped = df_selected.groupby(['Year', 'Month']).agg({'Quantity': 'sum'}).reset_index()

# Duomenys ruošiami modeliui
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_grouped['Quantity'].values.reshape(-1, 1))

# Mokymo ir testavimo rinkiniai
look_back = 12
X, Y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i, 0])
    Y.append(scaled_data[i, 0])

X, Y = np.array(X), np.array(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# LSTM modelis
def create_model(lstm_units=50, optimizer='adam'):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(lstm_units),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

param = {
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [16, 32, 64, 128],
    'epochs': sp_randint(10, 100),
    'lstm_units': sp_randint(50, 200)
}

# Inicijuojama KerasRegressor
keras_model = KerasRegressor(build_fn=create_model, lstm_units=133, verbose=1)

# Atsitiktinė paieška
# cv5 kartus mokomas, n iter atsitiktiniu deriniu bandymai
random_search = RandomizedSearchCV(estimator=keras_model, param_distributions=param,
                                   n_iter=20, cv=5, verbose=2, random_state=42)

# Paleidžiama paieška
random_search.fit(X_train, Y_train)

# Geriausi parametrai
print("Geriausi hiperparametrai:", random_search.best_params_)

# Geriausi parametrai
best_model = random_search.best_estimator_
predictions = best_model.predict(X_test)
rmse = np.sqrt(np.mean((predictions - Y_test) ** 2))
r2 = best_model.score(X_test, Y_test)
print("RMSE:", rmse)
print("R2:", r2)