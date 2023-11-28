# Importation des modules necessaires
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)


model.fit(X_train, y_train)

#Test du modèle
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
