import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, Activation
import numpy as np
from keras import regularizers
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)

train_X = np.load("/home/gerald/Desktop/heartdata/NNUCIDataset/train_data.npy")
train_label = np.load("/home/gerald/Desktop/heartdata/NNUCIDataset/train_label.npy")
test_X = np.load("/home/gerald/Desktop/heartdata/NNUCIDataset/test_data.npy")
test_label = np.load("/home/gerald/Desktop/heartdata/NNUCIDataset/test_label.npy")

def createmodel(numfirst = 64, numsecond = 40, regularizervalue = 0.01):
    model = tf.keras.Sequential()
    model.add(layers.Dense(numfirst, activation='relu', input_shape=(12,),kernel_initializer='random_uniform',
                    bias_initializer='zeros' ,kernel_regularizer=regularizers.l2(regularizervalue)))
    model.add(layers.Dense(numsecond, kernel_regularizer=regularizers.l2(regularizervalue), kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
    return model

def kfoldcrossval(data, data_label, kerasmodel, numepoches= 300, sizeofbatch= 40, kvalue = 5):
    kf = KFold(n_splits = kvalue)
    kf.get_n_splits(data)
    result = []
    for train_index, test_index in kf.split(data):
        X_train , X_test = data[train_index] , data[test_index]
        y_train , y_test = data_label[train_index], data_label[test_index]
        kerasmodel.fit(X_train, y_train, epochs = numepoches, batch_size = sizeofbatch)
        prediction = kerasmodel.predict(X_test) 
        prediction = np.where(prediction > 0.5, 1, 0)
        result.append(accuracy_score(y_test, prediction))
    return sum(result) / len(result)

def tunefirstlayer(firstlayer):
    result_layer = []
    for item in firstlayer:
        model = createmodel(numfirst = item)
        result = kfoldcrossval(train_X, train_label,model)
        result_layer.append(result)
    indexmax = np.argmax(result_layer)
    return firstlayer[indexmax] 

def tunesecondlayer(bestinfirst, secondlayer):
    result_layer = []
    for item in secondlayer:
        model = createmodel(numfirst = bestinfirst, numsecond = item)
        result = kfoldcrossval(train_X, train_label, model)
        result_layer.append(result)
    indexmax = np.argmax(result_layer)
    return secondlayer[indexmax]
def tunenumepoches(bestinfirst, bestinsecond, numepoches):
    result_layer = []
    for item in numepoches:
        model = createmodel(numfirst = bestinfirst, numsecond = bestinsecond)
        result = kfoldcrossval(train_X, train_label, model, numepoches= item)
        result_layer.append(result)
    indexmax = np.argmax(result_layer)
    return numepoches[indexmax]
def tunesizebatch(bestinfirst, bestinsecond, bestepoch, sizeofbatch):
    result_layer = []
    for item in sizeofbatch:
        model = createmodel(numfirst = bestinfirst, numsecond = bestinsecond)
        result = kfoldcrossval(train_X, train_label,model ,numepoches = bestepoch)
        result_layer.append(result)
    indexmax = np.argmax(result_layer)
    return sizeofbatch[indexmax]

def tune(firstlayer, secondlayer, numepoches, sizeofbatch):
    bestfirst = tunefirstlayer(firstlayer)
    bestsecond = tunesecondlayer(bestfirst, secondlayer)
    bestepoch = tunenumepoches(bestfirst, bestsecond, numepoches)
    bestbatch = tunesizebatch(bestfirst, bestsecond, bestepoch, sizeofbatch)
    return [bestfirst, bestsecond, bestepoch, bestbatch]
if __name__ == '__main__':
    firstlayer = [40,50,60,70,80,90,200]
    secondlayer = [100,90,80,70,60,50,40,30,20]
    numepoches = [500,400,300,200,100] 
    sizeofbatch = [100,90,80,70,60,50,40,30,20,10]
    param = tune(firstlayer, secondlayer, numepoches, sizeofbatch)
    print("Parameters: " + str(param))
    model = createmodel(param[0], param[1])
    history = model.fit(train_X, train_label, epochs = param[2], batch_size = param[3])
    prediction = model.predict(test_X)
    prediction = np.where(prediction > 0.5, 1, 0)
    result_accuracy = accuracy_score(test_label, prediction)
    print("Result Accuracy: " + str(result_accuracy))
    f = open("/home/gerald/Desktop/heartdata/NNUCIDataset/testaccuracy.txt/accuracy.txt", "a")
    f.write(str(result_accuracy))
    f.write(str(param))
    f.close()
    model.save('finalmodel.h5')
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
