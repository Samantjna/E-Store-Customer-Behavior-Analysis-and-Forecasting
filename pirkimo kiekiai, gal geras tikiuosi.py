import dash
from dash import dcc, html
import plotly.graph_objs as go
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
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')

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

# Dash programėlė
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=list(range(len(y_test))),
                    y=y_test,
                    mode='markers',
                    name='Tikrosios reikšmės',
                    marker=dict(color='blue')
                ),
                go.Scatter(
                    x=list(range(len(y_pred))),
                    y=y_pred.flatten(),
                    mode='markers',
                    name='Prognozuojamos reikšmės',
                    marker=dict(color='red')
                )
            ],
            'layout': go.Layout(
                title='Tikrosios ir prognozuojamos reikšmės',
                xaxis=dict(title='Indeksas'),
                yaxis=dict(title='Kiekis'),
                legend=dict(x=0, y=1.0),
                hovermode='closest'
            )
        }
    ),
    html.Div([
        html.H4(f'R2 rezultatas: {r2}'),
        html.H4(f'MSE rezultatas: {mse}'),
        html.H4(f'MAE rezultatas: {mae}')
    ])
])
#po paleidimo reikia sustabdyti, nes leidžia procesą iš naujo, kad gautų vis naujus duomenis
if __name__ == '__main__':
    app.run_server()