import numpy as np
import math
import pandas as pd

train_data = np.load("train_data.npy")


train_data_columns = train_data.T

age_mean = np.average(train_data_columns[0])
age_sd = np.std(train_data_columns[0])

trestbps_mean = np.average(train_data_columns[2])
trestbps_sd = np.std(train_data_columns[2])

chol_mean =  np.average(train_data_columns[3])
chol_sd = np.std(train_data_columns[3])

thalach_mean =  np.average(train_data_columns[6])
thalach_sd = np.std(train_data_columns[6])


for i in range(len(train_data_columns)):
    # normalizing age
    if i == 0:
        normalized_ages = []
        for j in range(len(train_data_columns[i])):
            age = train_data_columns[i][j]
            normalized_age = (age - age_mean)/age_sd
            normalized_ages.append(normalized_age)
        train_data_columns[0] = normalized_ages
    # normalizing trestbps
    if i == 2:
        normalized_trestbps = []
        for j in range(len(train_data_columns[i])):
            trestbps = train_data_columns[i][j]
            normalizedtrestbps = (trestbps - trestbps_mean)/trestbps_sd
            normalized_trestbps.append(normalizedtrestbps)
        train_data_columns[2] = normalized_trestbps
    # normalizing chol
    if i == 3:
        normalized_chols = []
        for j in range(len(train_data_columns[i])):
            chol = train_data_columns[i][j]
            normalized_chol = (chol - chol_mean) / chol_sd
            normalized_chols.append(normalized_chol)
        train_data_columns[3] = normalized_chols
    # normalizing thalach
    if i == 6:
        normalized_thalachs = []
        for j in range(len(train_data_columns[i])):
            thalach = train_data_columns[i][j]
            normalized_thalach = (thalach - thalach_mean) / thalach_sd
            normalized_thalachs.append(normalized_thalach)
        train_data_columns[6] = normalized_thalachs


print(train_data_columns)
np.save("normalized_train_data", train_data_columns.T)



test_data = np.load("test_data.npy")


test_data_columns = test_data.T

age_mean = np.average(test_data_columns[0])
age_sd = np.std(test_data_columns[0])

trestbps_mean = np.average(test_data_columns[2])
trestbps_sd = np.std(test_data_columns[2])

chol_mean =  np.average(test_data_columns[3])
chol_sd = np.std(test_data_columns[3])

thalach_mean =  np.average(test_data_columns[6])
thalach_sd = np.std(test_data_columns[6])


for i in range(len(test_data_columns)):
    # normalizing age
    if i == 0:
        normalized_ages = []
        for j in range(len(test_data_columns[i])):
            age = test_data_columns[i][j]
            normalized_age = (age - age_mean)/age_sd
            normalized_ages.append(normalized_age)
        test_data_columns[0] = normalized_ages
    # normalizing trestbps
    if i == 2:
        normalized_trestbps = []
        for j in range(len(test_data_columns[i])):
            trestbps = test_data_columns[i][j]
            normalizedtrestbps = (trestbps - trestbps_mean)/trestbps_sd
            normalized_trestbps.append(normalizedtrestbps)
        test_data_columns[2] = normalized_trestbps
    # normalizing chol
    if i == 3:
        normalized_chols = []
        for j in range(len(test_data_columns[i])):
            chol = test_data_columns[i][j]
            normalized_chol = (chol - chol_mean) / chol_sd
            normalized_chols.append(normalized_chol)
        test_data_columns[3] = normalized_chols
    # normalizing thalach
    if i == 6:
        normalized_thalachs = []
        for j in range(len(test_data_columns[i])):
            thalach = test_data_columns[i][j]
            normalized_thalach = (thalach - thalach_mean) / thalach_sd
            normalized_thalachs.append(normalized_thalach)
        test_data_columns[6] = normalized_thalachs


print(test_data_columns)
np.save("normalized_test_data", test_data_columns.T)
