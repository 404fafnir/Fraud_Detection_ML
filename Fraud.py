from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

csv_path = 'Dataset.csv'
target_column = 'isFraud'

def preprocess_csv(csv_path, target_column=None, numeric_features=None, categorical_features=None):
    """
    Prétraite un fichier CSV en effectuant des opérations courantes.

    Paramètres :
    - csv_path (str) : Chemin vers le fichier CSV.
    - target_column (str) : Nom de la colonne cible (variable dépendante). Si None, aucune colonne cible n'est traitée.
    - numeric_features (list) : Liste des noms de colonnes numériques.
    - categorical_features (list) : Liste des noms de colonnes catégorielles.

    Retourne :
    - X (DataFrame) : Caractéristiques prétraitées.
    - y (DataFrame) : Étiquettes (si target_column n'est pas None).
    """
    # Charger le CSV en tant que DataFrame
    df = pd.read_csv(csv_path)

    # Séparer les caractéristiques (X) et les étiquettes (y)
    if target_column is not None:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df

    # Prétraitement des caractéristiques numériques
    if numeric_features is not None:
        numeric_transformer = StandardScaler()
        X[numeric_features] = numeric_transformer.fit_transform(X[numeric_features])

    # Prétraitement des caractéristiques catégorielles
    if categorical_features is not None:
        categorical_transformer = LabelEncoder()
        X[categorical_features] = X[categorical_features].apply(lambda col: categorical_transformer.fit_transform(col.astype(str)))

        print(df.info)

    # Retourner les données prétraitées
    if target_column is not None:
        return X, y
    else:
        return X

# Exemple d'utilisation

numeric_features = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud']
categorical_features = ['type','nameOrig','nameDest']

X_preprocessed, y_preprocessed = preprocess_csv(csv_path, target_column, numeric_features, categorical_features)



def split_csv(csv_path, target_column, test_size=0.2, random_state=42):
    """
    Divise un fichier CSV en ensembles d'entraînement et de test.

    Paramètres :
    - csv_path (str) : Chemin vers le fichier CSV.
    - target_column (str) : Nom de la colonne cible (variable dépendante).
    - test_size (float) : Proportion des données à inclure dans l'ensemble de test.
    - random_state (int) : Graine aléatoire pour la reproductibilité.

    Retourne :
    - X_train (DataFrame) : Caractéristiques de l'ensemble d'entraînement.
    - X_test (DataFrame) : Caractéristiques de l'ensemble de test.
    - y_train (DataFrame) : Étiquettes de l'ensemble d'entraînement.
    - y_test (DataFrame) : Étiquettes de l'ensemble de test.
    """
    # Charger le CSV en tant que DataFrame
    df = pd.read_csv(csv_path)

    # Séparer les caractéristiques (X) et les étiquettes (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# Exemple d'utilisation

X_train, X_test, y_train, y_test = split_csv(csv_path, target_column)


clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))