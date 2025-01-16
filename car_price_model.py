import pandas as pd
from keras.layers import Dropout
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Wczytanie danych
data = pd.read_csv('car_features_and_msrp.csv')

# Usunięcie duplikatów i braków danych
data = data.drop_duplicates()
data = data.dropna()

# Usunięcie niepotrzebnych kolumn
data = data.drop(['Engine Fuel Type', 'Market Category', 'Vehicle Style',
                  'Popularity', 'Number of Doors', 'Vehicle Size'], axis=1)

# Usunięcie anomalii w 'Transmission Type'
data.drop(data[data['Transmission Type'] == 'UNKNOWN'].index, axis='index', inplace=True)

# Usunięcie odstających wartości (Samochody o cenie powyżej 300 000 USD)
data.drop(data[data['MSRP'] >= 300000].index, inplace=True)

# Wykres rozkładu ilości samochodów według marki
# data.Make.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
# plt.title('Rozkład ilości samochodów według marki')
# plt.ylabel('Ilość')
# plt.xlabel('Marka')
# plt.show()

# Kodowanie zmiennych kategorycznych za pomocą one-hot encoding
cat_features = ['Make', 'Model', 'Transmission Type', 'Driven_Wheels']
data = pd.get_dummies(data, columns=cat_features)


# Podział na cechy (X) i zmienną docelową (y)
X = data.drop('MSRP', axis=1)
y = data['MSRP']



# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Skalowanie danych
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))  # Regularyzacja Dropout
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Trening modelu
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# Ewaluacja modelu na danych testowych
loss, mae = model.evaluate(X_test, y_test, verbose=0)

# Predykcja na danych testowych
y_pred = model.predict(X_test)

# Wyświetlenie przykładowych wyników
print("Przykładowe wyniki:")
for i in range(5):
    print(f"Faktyczna cena: {y_test.iloc[i]} USD, Przewidywana: {y_pred[i][0]:.2f} USD")


# Sprawdzenie, czy model osiąga dobry wynik na podstawie wartości MAE
# Obliczenie średniej ceny w zbiorze danych
mean_price = data['MSRP'].mean()

# Obliczenie 10% średniej ceny
threshold = 0.1 * mean_price

# Porównanie MAE z progiem
print(f"Średnia cena (mean MSRP): {mean_price:.2f}")
print(f"10% średniej ceny: {threshold:.2f}")
print(f"MAE: {mae:.2f}")

if mae < threshold:
    print("Model osiąga dobry wynik, ponieważ MAE jest mniejsze niż 10% średniej ceny.")
else:
    print("Model można ulepszyć, ponieważ MAE przekracza 10% średniej ceny.")


# Zapis modelu
model.save('car_price_prediction_model.h5')