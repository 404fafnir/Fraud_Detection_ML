from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import csv

# Parse the dataset to have lists to work on

step = []
type = []
amount = []
nameOrig = []
oldBalanceOrg = []
newBalanceOrg = []
nameDest = []
oldBalanceDest = []
newBalanceDest = []
isFraud = []
isFlaggedFraud = []

with open('Dataset.csv', newline='') as CSVdataset:
    spamreader = csv.reader(CSVdataset, delimiter=',')
    for row in spamreader:
        step.append(row[0])
        type.append(row[1])
        amount.append(row[2])
        nameOrig.append(row[3])
        oldBalanceOrg.append(row[4])
        newBalanceOrg.append(row[5])
        nameDest.append(row[6])
        oldBalanceDest.append(row[7])
        newBalanceDest.append(row[8])
        isFraud.append(row[9])
        isFlaggedFraud.append(row[10])


neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(step,type, amount, nameOrig, oldBalanceOrg, newBalanceOrg, nameDest, oldBalanceDest, newBalanceDest, isFraud, isFlaggedFraud)


