# Importation des modules necessaires
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import mean_squared_error

#Conversion du Dataset en DataFrame
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

#Lasso avec un alpha
setAlpha = 0.5

lasso_model = linear_model.Lasso(alpha=setAlpha)
lasso_model.fit(X_train, y_train)

#Test du modèle
predictions = lasso_model.predict(X_test)

#Calcul du taux d'erreur MSE
mse = mean_squared_error(y_test, predictions)
print("alpha égale à :" , setAlpha)
print("Erreur quadratique moyenne du Modèle Lasso:", mse)
print("Précision :", 1-mse)
