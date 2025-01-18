import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow.keras.backend as K

# Funkcja straty kwantylowej
def quantile_loss(q, y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    error = y_true - y_pred
    return K.mean(K.maximum(q * error, (q - 1) * error), axis=-1)

# ETAP Wczytanie datasetu
df = pd.read_csv("data/House_Rent_Dataset.csv")

# Analiza danych zbioru
print("Podstawowe informacje o danych:")
print(df.info())
print("\nPodstawowe statystyki:")
print(df.describe())

sns.histplot(df['Rent'], kde=True)
plt.title("Rozkład cen wynajmu")
plt.show()

# Obsłużenie braku danych
print("\nBraki danych w kolumnach:")
print(df.isnull().sum())
# Uzupełnienie średnią
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Przetwarzanie kolumny 'Floor'
def preprocess_floor(floor_value):
    if 'Ground' in floor_value:
        return 0
    try:
        return int(floor_value.split(' ')[0])
    except ValueError:
        return np.nan

df['Floor'] = df['Floor'].apply(preprocess_floor)
df['Floor'] = df['Floor'].fillna(df['Floor'].median())

# Usuwanie odstających wartości korzystając z analizy rozkładu IQR która pozwala na eliminację skrajnych wartości, które mogą zniekształcać model.
sns.boxplot(x=df['Rent'])
plt.title("Wykrywanie wartości odstających dla cen wynajmu")
plt.show()

Q1 = df['Rent'].quantile(0.25)
Q3 = df['Rent'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Rent'] >= lower_bound) & (df['Rent'] <= upper_bound)]

# Zakodowanie zmiennych kategorycznych - uwzględnienie informacji o kategoriach w modelu numerycznym.
df = pd.get_dummies(df, columns=['Area Locality', 'Area Type', 'City', 'Furnishing Status', 'Tenant Preferred'], drop_first=True)

# Usunięcie kolumny 'Point of Contact'
df.drop(columns=['Point of Contact'], inplace=True)#  (nienumeryczna i nieistotna dla analizy)

# Weryfikacja kolumn liczbowych
non_numeric_columns = df.select_dtypes(include=['object']).columns
if len(non_numeric_columns) > 0:
    print(f"\nUsuwane kolumny nienumeryczne przed standaryzacją: {non_numeric_columns.tolist()}")
    df.drop(columns=non_numeric_columns, inplace=True)


# Zapisanie kolumn przetworzonych
processed_columns = df.drop('Rent', axis=1).columns.tolist()
with open("processed_columns.txt", "w") as f:
    for col in processed_columns:
        f.write(col + "\n")

# Przygotowanie danych
def prepare_data(df):
    X = df.drop('Rent', axis=1)
    y = df['Rent']

    # Skalowanie danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Zapisanie scalera do pliku
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Podzielenie na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = prepare_data(df)

# Zredukowanie wymiarów za pomocą PCA
# Redukcja liczb cech przy zachowaniu wiekszosci warjancji
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_train)

print(f"\nLiczba komponentów zachowujących 95% wariancji: {pca.n_components_}")
explained_variance = pca.explained_variance_ratio_
print("\nWyjaśniona wariancja przez komponenty PCA:")
print(explained_variance)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.title('Skumulowana wariancja wyjaśniona przez komponenty PCA')
plt.xlabel('Liczba komponentów')
plt.ylabel('Skumulowana wariancja')
plt.grid()
plt.show()

# Definicja kwantyli
quantiles = [0.1, 0.5, 0.9]

# Trenowanie modeli
models = {}
for q in quantiles:
    print(f"Trenowanie modelu dla kwantyla: {q}")
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])

    # Kompilacja modelu
    model.compile(optimizer=Adam(learning_rate=0.001), loss=lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    # Trenowanie
    history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

    model.save(f"models/quantile_model_{int(q*100)}.h5")
    models[q] = model

    # Wizualizacja strat
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Strata treningowa')
    plt.plot(history.history['val_loss'], label='Strata walidacyjna')
    plt.title(f'Krzywa uczenia modelu dla kwantyla {q}')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    plt.grid()
    plt.show()

# Ewaluacja modeli
for q, model in models.items():
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nEwaluacja dla kwantyla {q}:")
    print(f"MAE: {mae:.2f}")

# Zapisanie predykcji do pliku
predictions = {f"Quantile_{q}": model.predict(X_test).flatten() for q, model in models.items()}
predictions['True Values'] = y_test.values
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("quantile_predictions.csv", index=False)
print("\nPredykcje zapisane do pliku quantile_predictions.csv")