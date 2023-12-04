# Importation des modules necessaires
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

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

#KNN sur les 3 plus proches voisins
clustering = AgglomerativeClustering(n_clusters=3)
clustering.fit(X_train)

#Test du modèle


#Calcul du taux d'erreur MSE

plt.scatter(X_train[:, 0], X_train[:, 1], c=model.labels_, cmap='rainbow')
plt.title('Clustering hiérarchique agglomératif')
plt.show()

linked = linkage(X, 'ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.show()