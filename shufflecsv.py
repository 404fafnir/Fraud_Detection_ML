import pandas as pd
import random

# step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud

def melanger_lignes_csv(input_file, output_file):
    # Charger le fichier CSV
    data = pd.read_csv(input_file)

    # Mélanger les lignes
    data_shuffled = data.sample(frac=1, random_state=random.seed())  # frac=1 signifie utiliser toutes les lignes

    # Enregistrer le résultat dans un nouveau fichier CSV
    data_shuffled.to_csv(output_file, index=False)

# Exemple d'utilisation
input_file = 'Dataset.csv'
output_file = 'fichier_melange.csv'
melanger_lignes_csv(input_file, output_file)