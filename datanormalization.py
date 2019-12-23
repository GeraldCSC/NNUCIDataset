import numpy as np
import pandas as pd

train_data = np.load("train_data.npy")
test_data = np.load("test_data.npy")
print(train_data)
def normalizecol(colarray):
    mean = np.mean(colarray)
    sd = np.std(colarray)
    colarray = (colarray - mean) / sd
    print(colarray.shape)
    return colarray.flatten()
indices_to_standardize = [0,2,3,6] 
for item in indices_to_standardize:
    train_data[:,item] = normalizecol(train_data[:,item])    
    test_data[:,item] = normalizecol(test_data[:,item])
print(train_data)
print(test_data)
np.save("normalized_train_data", train_data)
np.save("normalized_test_data", test_data)
