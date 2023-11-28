from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import mean_squared_error


data = pd.read_csv('Dataset.csv')

label_encoder = LabelEncoder()

data['type'] = label_encoder.fit_transform(data['type'])
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])

X = data.drop('isFraud', axis=1) 
y = data['isFraud']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)


neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(X_train, y_train)

predictions = neigh.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("Erreur quadratique moyenne du modèle k-NN:", mse)
