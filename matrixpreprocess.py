import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def getdata(path):
    data = pd.read_csv(path) 
    #we treat sex as a protected variable, so it is deleted from our features
    del data['sex']
    labels = data['target'].to_numpy()
    del data['target'] 
    data_matrix = data.to_numpy()
    return data_matrix , labels

def get_train_test_split(data_matrix, labels, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        data_matrix, labels, test_size=0.33, random_state=42)
    return X_train, y_train, X_test, y_test

def _normalizecol(colarray):
    mean = np.mean(colarray)
    sd = np.std(colarray)
    colarray = (colarray - mean) / sd
    return colarray.flatten()

def normalizematrix(X_train, X_test,indices_to_norm):
    for item in indices_to_norm:
        X_train[:,item] = _normalizecol(X_train[:,item])
        X_test[:,item] = _normalizecol(X_test[:,item])
    return X_train, X_test

def get_normalized_data_set(path, test_size, indices_to_norm):
    data_matrix, labels = getdata(path)
    X_train, y_train, X_test, y_test = get_train_test_split(data_matrix, labels, test_size)
    X_train, X_test = normalizematrix(X_train, X_test, indices_to_norm)
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    pass
