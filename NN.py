import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from matrixpreprocess import get_normalized_data_set
from matplotlib import pyplot as plt

def getmodel(numfirst = 200, numsecond = 100, reg_value = 0.01):
    model = keras.Sequential([
    Dense(numfirst, activation='relu',
                 kernel_regularizer=l2(l=reg_value), input_shape = (12,)),
    Dropout(0.5),
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

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_normalized_data_set("heart.csv",0.33,[0,2,3,6])
    model = getmodel()
    accuracy = train_and_eval(model, X_train, y_train, X_test, y_test, 500)
    print(accuracy)
