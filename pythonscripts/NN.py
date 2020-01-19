import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from matrixpreprocess import get_normalized_data_set
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

def getmodel(numfirst = 200, numsecond = 100, reg_value = 0.01, dropoutratelayer1 = 0.2, 
             dropoutratelayer2=0.2):
    model = keras.Sequential([
    Dense(numfirst, activation='relu',
                 kernel_regularizer=l2(l=reg_value), input_shape = (12,)),
    Dropout(dropoutratelayer1),
    Dense(numsecond, activation='relu',kernel_regularizer=l2(l=reg_value)),
    Dropout(dropoutratelayer2), 
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l=reg_value))])
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model


def plot(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

def train_and_eval(model, X_train, y_train, X_eval, y_eval, numepoch):
    history = model.fit(X_train, y_train, epochs = numepoch,verbose = 0)
    return evaluate(model,X_eval, y_eval)

def evaluate(model,X_eval,y_eval):
    history_eval = model.evaluate(X_eval, y_eval)
    return history_eval[1]

def kfoldcrossval(data,data_label,model, numepochs= 500,k = 5):
    kf = KFold(n_splits = k)
    kf.get_n_splits(data)
    result = []
    for train_index, val_index in kf.split(data):
        X_train , X_val = data[train_index] , data[val_index]
        y_train , y_val = data_label[train_index], data_label[val_index]
        accuracy = train_and_eval(model, X_train, y_train, X_val, y_val, numepochs)
        result.append(accuracy)
    return sum(result) / len(result)

def getbestmodel(X_train, y_train, epochs,reg_rate):
    maxacc = 0
    for reg in reg_rate:
        for e in epochs: 
            model = getmodel(reg_value = reg)
            mean_acc = kfoldcrossval(X_train, y_train, model,e)
            if maxacc < mean_acc: 
                bestmodel = model
                reg_to_use = reg
                epoch_to_use = e
                maxacc = mean_acc
    return reg_to_use, epoch_to_use,maxacc, bestmodel

if __name__ == "__main__":
    X_train, y_train, X_test, y_test =get_normalized_data_set("/Users/gerald/Documents/GitHub/NNUCIDataset/heart.csv",0.33,[0,2,3,6])
    regvalue , epochvalue,train_accuracy,model = getbestmodel(X_train, y_train, [500,1000],[0,0.001,0.01,0.1])
    accuracy = evaluate(model, X_test,y_test)
    print(accuracy)
