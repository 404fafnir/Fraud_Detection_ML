import pandas as pd


def clean_csv(input_file, output_file):
    # Charger le fichier CSV
    data = pd.read_csv(input_file)

    colones = ["step", "nameOrig", "nameDest", "isFlaggedFraud"]

    data_clean = data.drop(colones, axis=1)

    # Enregistrer le résultat dans un nouveau fichier CSV
    data_clean.to_csv(output_file, index=False)

# Exemple d'utilisation
input_file = 'Dataset.csv'
output_file = 'fichier_clean.csv'
clean_csv(input_file, output_file)