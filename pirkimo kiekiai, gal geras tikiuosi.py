import plotly.express as px
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

# Duomenų įkėlimas
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

# Standartizuojami duomenys
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Padalinama į mokymo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelis
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])

# Apmokymas
history = model.fit(X_train, y_train, epochs=32, batch_size=49, validation_split=0.2)

# Prognozės
y_pred = model.predict(X_test)

# Vertinimai
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# fig = px.scatter(df, x=y_test, y=y_pred, title='Faktinis skaičius vs. prognozuojamas skaičius')
# fig.update_traces(marker=dict(line=dict(width=1, color='rgba(0, 255, 0, 0)')), selector=dict(mode='markers'))
# fig.show()



import plotly.express as px

# Sukurkite DataFrame su tikraisiais ir prognozuojamais kiekiais bei jų atitinkamomis datos reikšmėmis
results_df = pd.DataFrame({
    'Date': X_test[:, -1],  # Čia prielaidos, kad paskutinis stulpelis yra datos
    'Actual': y_test,
    'Predicted': y_pred.flatten()
})

# Konvertuokite laiko žymas atgal į datą
results_df['Date'] = pd.to_datetime(results_df['Date'], unit='s')

# Vizualizacija su Line plot
fig = px.line(results_df, x='Date', y=['Actual', 'Predicted'], title='Faktinės ir Prognozuojamos Reikšmės per Laiką')
fig.show()