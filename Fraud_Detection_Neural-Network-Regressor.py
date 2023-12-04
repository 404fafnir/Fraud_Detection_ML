# Importation des modules necessaires
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Dataset.csv')

#Initialisationd de la fonction LabelEncoder()
label_encoder = LabelEncoder()

#Conversion des données catégoricielles en données numériques utilisable par le modèle
data['type'] = label_encoder.fit_transform(data['type'])
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])

#Definiton de la donée cible
X = data.drop('isFraud', axis=1) 
y = data['isFraud']

#Division du Dataset en un ensemble de d'entrainement et un ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Reseau de Neurone
model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

#Entrainement du réseau de Neurone
model.fit(X_train, y_train)

#Test du modèle
predictions = model.predict(X_test)

#Calcul du taux d'erreur
mse = mean_squared_error(y_test, predictions)

print("Le Meqn Square Error est : ", mse)
print("Accuracy : ", 1-mse)