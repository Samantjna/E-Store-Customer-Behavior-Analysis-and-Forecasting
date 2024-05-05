import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt

# Įkeliam duomenis
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv', nrows=20000)

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

# Modelio kūrimo funkcija
def build_model(optimizer='adam', units=64):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(units // 2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    return model

# KerasRegressor sukuria modelio įvilką naudojant scikit-learn
model = KerasRegressor(build_fn=build_model, units=128)

# Hiperparametrų tinklelis
param_dist = {
    'optimizer': ['adam', 'rmsprop'],
    'epochs': sp_randint(20, 50),
    'batch_size': sp_randint(32, 64)
}

# RandomizedSearchCV sukūrimas
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5, scoring='neg_mean_squared_error', cv=2)

# Hiperparametrų paieška
random_search_result = random_search.fit(X_train, y_train)

# Geriausių parametrų atspausdinimas
print("Geriausias rezultatas: %f naudojant %s" % (random_search_result.best_score_, random_search_result.best_params_))

# Modelio vertinimas su geriausiais parametrais
best_model = build_model(
    optimizer=random_search_result.best_params_['optimizer'],
    units=random_search_result.best_params_['units']
)
best_model.fit(
    X_train,
    y_train,
    epochs=random_search_result.best_params_['epochs'],
    batch_size=random_search_result.best_params_['batch_size']
)

# Modelio prognozės
y_pred = best_model.predict(X_test)

# R^2
r2 = r2_score(y_test, y_pred)
print("R^2:", r2)

# MAE
mae = mean_absolute_error(y_test, y_pred)
print("Vidutinis absoliutus nuokrypis (MAE):", mae)

