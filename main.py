import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
#from tensorflow.keras.utils import to_categorical

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

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Usunięcie pewnych kolumn
threshold = 0.5
missing_percentage = df.isnull().mean()
columns_to_drop = missing_percentage[missing_percentage > threshold].index
df.drop(columns=columns_to_drop, inplace=True)
print("\nKolumny usunięte z powodu dużej liczby braków:", columns_to_drop.tolist())

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

# Usuwanie odstających wartości
sns.boxplot(x=df['Rent'])
plt.title("Wykrywanie wartości odstających dla cen wynajmu")
plt.show()

Q1 = df['Rent'].quantile(0.25)
Q3 = df['Rent'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Rent'] >= lower_bound) & (df['Rent'] <= upper_bound)]

# Zakodowanie zmiennych kategorycznych
df = pd.get_dummies(df, columns=['Area Locality', 'Area Type', 'City', 'Furnishing Status', 'Tenant Preferred'], drop_first=True)

# Usunięcie kolumny 'Point of Contact'
df.drop(columns=['Point of Contact'], inplace=True)#  (nienumeryczna i nieistotna dla analizy)

# Weryfikacja kolumn liczbowych
non_numeric_columns = df.select_dtypes(include=['object']).columns
if len(non_numeric_columns) > 0:
    print(f"\nUsuwane kolumny nienumeryczne przed standaryzacją: {non_numeric_columns.tolist()}")
    df.drop(columns=non_numeric_columns, inplace=True)

# Analizowanie korelacji
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_df = df[numeric_columns]

correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Macierz korelacji")
plt.show()

correlation_with_rent = correlation_matrix['Rent'].sort_values(ascending=False)
print("\nKorelacja zmiennych z 'Rent':")
print(correlation_with_rent)

low_correlation_columns = correlation_with_rent[correlation_with_rent.abs() < 0.1].index
df.drop(columns=low_correlation_columns, inplace=True)
print("\nUsunięte zmienne o niskiej korelacji:", low_correlation_columns.tolist())

# Zapisanie kolumn przetworzonych
processed_columns = df.drop('Rent', axis=1).columns.tolist()
with open("processed_columns.txt", "w") as f:
    for col in processed_columns:
        f.write(col + "\n")

# Przygotowanie danych
X = df.drop('Rent', axis=1)
y = df['Rent']

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Zapisanie scalera do pliku
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Podzielenie na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Zredukowanie wymiarów za pomocą PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

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

# ETAP Budowa modelu regresji**
# Przygotowanie danych
X = df.drop('Rent', axis=1)
y = df['Rent']

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podzielenie danych na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Budowanie modelu
model = Sequential([
    #Input(shape=(X_train.shape[1],)),
    Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer='l2'),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer='l2'),
    #Dropout(0.2),
    Dense(1, activation='linear')
])

# Kompilowanie modelu
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

early_stopping_1 = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr_1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping_1, reduce_lr_1])


# Ewaluowanie modelu
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("\nEwaluacja modelu Keras:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Zapisanie model regresji
model.save("models/regression_model.h5")

# Wizualizacja wyników
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.title('Krzywa uczenia modelu Keras')
plt.xlabel('Epoka')
plt.ylabel('Strata (MSE)')
plt.legend()
plt.grid()
plt.show()

# ETAP Budowa modelu klasyfikacji
# Określenie progu
median_rent = y.median()
df['Rent_Class'] = (y > median_rent).astype(int)  # 0: Tanie mieszkanie, 1: Drogie mieszkanie

# Przygotowanie danych
X = df.drop(['Rent', 'Rent_Class'], axis=1)
y_class = df['Rent_Class']

# Podzielenie na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

y_train = np.array(y_train).astype('float32')
y_test = np.array(y_test).astype('float32')

# Budowanie modelu klasyfikacji
model_classification = Sequential([
    #Input(shape=(X_train.shape[1],)),
    Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'),
    Dense(64, activation='relu', kernel_regularizer='l2'),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer='l2'),
    Dense(1, activation='sigmoid')
])

# Kompilowanie modelu
model_classification.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

early_stopping_2 = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
reduce_lr_2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Trenowanie modelu
history_classification = model_classification.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping_2, reduce_lr_2])

# Ewaluowanie modelu
y_pred_probs = model_classification.predict(X_test).flatten()
y_pred_classes = (y_pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)
roc_auc = roc_auc_score(y_test, y_pred_probs)

print("\nEwaluacja modelu klasyfikacji Keras:")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

# Zapisanie model
model_classification.save("models/classification_model.h5")

# Wizualizacja wyników klasyfikacji
plt.figure(figsize=(10, 5))
plt.plot(history_classification.history['loss'], label='Strata treningowa')
plt.plot(history_classification.history['val_loss'], label='Strata walidacyjna')
plt.title('Krzywa uczenia modelu klasyfikacji Keras')
plt.xlabel('Epoka')
plt.ylabel('Strata (Binary Crossentropy)')
plt.legend()
plt.grid()
plt.show()

# Wykres dokładności
plt.figure(figsize=(10, 5))
plt.plot(history_classification.history['accuracy'], label='Dokładność treningowa')
plt.plot(history_classification.history['val_accuracy'], label='Dokładność walidacyjna')
plt.title('Dokładność modelu klasyfikacji Keras')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid()
plt.show()

# Wykres predykcji kontra rzeczywiste wartości
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Predykcje vs Rzeczywiste wartości')
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Predykcje')
plt.grid()
plt.show()