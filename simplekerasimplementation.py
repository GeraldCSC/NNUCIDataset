import keras
from keras import Sequential
from keras import Dense
import numpy as np

train_data = np.load("normalized_train_data.npy")
train_label = np.load("train_label.npy")
test_data = np.load("normalized_test_data.npy")
test_label = np.load("test_label.npy")
