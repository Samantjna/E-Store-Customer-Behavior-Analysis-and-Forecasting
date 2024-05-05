import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Sukuriam Dash aplikaciją
app = dash.Dash(__name__)

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


# Sukuriame DataFrame prognozėms
predictions_df = pd.DataFrame({
    'Purchase Date': X_test[:, 0],  # Čia priklauso nuo to, kaip jūsų duomenys yra išdėstyti
    'Predicted Quantity': y_pred.flatten()
})

# Sukuriam linijinį grafiką su Plotly
fig = px.line(predictions_df, x='Purchase Date', y='Predicted Quantity', title='Prognozuojamas pardavimų kiekis per laiką')

# Aplikacijos išdėstymas
app.layout = html.Div([
    html.H1('E-komercijos klientų elgsenos analizė ir prognozavimas'),
    html.P('Šiame grafike matome modelio prognozes, kaip keisis pardavimų kiekis per laiką.'),
    dcc.Graph(
        id='time-series-predictions',
        figure=fig
    ),
    html.P('Linijinis grafikas leidžia matyti pardavimų kiekio tendencijas ir prognozes. Tai padeda suprasti, kokių pokyčių galime tikėtis ateityje.'),
])

# Paleidžiam aplikaciją
if __name__ == '__main__':
    app.run_server()