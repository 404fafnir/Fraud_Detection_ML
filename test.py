# importing necessary libraries ( Dependencies)

# for numerical operations
import pandas as pd
import numpy as np

# for graphical visualization
import matplotlib.pyplot as plt
import seaborn as sns

Manish=pd.read_csv('Dataset.csv')

df = pd.get_dummies(Manish, columns=['type', 'nameOrig', 'nameDest'], prefix=['type', 'nameOrig', 'nameDest'])


#print(Manish.dtypes)
#print(Manish.head)