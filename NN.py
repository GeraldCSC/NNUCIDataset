import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from matrixpreprocess import get_normalized_data_set
from matplotlib import pyplot as plt

def getmodel(numfirst = 200, numsecond = 100, reg_value = 0.01, dropoutrate=0.5):
    model = keras.Sequential([
    Dense(numfirst, activation='relu',
                 kernel_regularizer=l2(l=reg_value), input_shape = (12,)),
    Dropout(dropoutrate),
    Dense(numsecond, activation='relu',kernel_regularizer=l2(l=reg_value)),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l=reg_value))])
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def train_and_eval(model, X_train, y_train, X_eval, y_eval, numepoch):
    model.fit(X_train, y_train, epochs = numepoch)
    history_eval = model.evaluate(X_eval, y_eval)
    return history_eval[1]

def kfoldcrossval(data,data_label,model, numepoches= 500,k = 5):
    kf = KFold(n_splits = kvalue)
    kf.get_n_splits(data)
    result = []
    for train_index, test_index in kf.split(data):
        X_train , X_test = data[train_index] , data[test_index]
        y_train , y_test = data_label[train_index], data_label[test_index]
        #predict and evaluate in this block here
    #return max accuracy and the model params/model here 

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_normalized_data_set("heart.csv",0.33,[0,2,3,6])
    model = getmodel()
    accuracy = train_and_eval(model, X_train, y_train, X_test, y_test, 500)
    print(accuracy)
