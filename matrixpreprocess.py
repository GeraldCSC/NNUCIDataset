import numpy as np
import pandas as pd
import math
data = pd.read_csv("heart.csv") 
del data['sex']
targets = data['target'].to_numpy()
del data['target'] 
array = data.to_numpy()
heart_attack_data = array[:165,:]
heart_attack_label = targets[:165]
no_heartattack_data = array[165:,:]
no_heartattack_label = targets[165:]
index_no_heart_attack = np.random.permutation(np.arange(no_heartattack_label.shape[0]))
index_yes_heart_attack = np.random.permutation(np.arange(heart_attack_label.shape[0]))
heart_attack_data = heart_attack_data[index_yes_heart_attack]
heart_attack_label = heart_attack_label[index_yes_heart_attack]
no_heartattack_data = no_heartattack_data[index_no_heart_attack]
no_heartattack_label = no_heartattack_label[index_no_heart_attack]
splitheartattack = math.ceil(heart_attack_label.shape[0] * 0.7)
split_noheartattack = math.ceil(no_heartattack_label.shape[0] * 0.7)
heartattacktrain , heartattacktrainlabel = heart_attack_data[:splitheartattack,:] , heart_attack_label[:splitheartattack]
heartattacktest , heartattacktestlabel = heart_attack_data[splitheartattack:,:], heart_attack_label[splitheartattack:]
noattacktrain , noattacktrainlabel = no_heartattack_data[:split_noheartattack,:] , no_heartattack_label[:split_noheartattack]
noattacktest , noattacktestlabel = no_heartattack_data[split_noheartattack:,:], no_heartattack_label[split_noheartattack:]
train = np.vstack((heartattacktrain, noattacktrain))
train_label = np.concatenate((heartattacktrainlabel, noattacktrainlabel))
test = np.vstack((heartattacktest, noattacktest))
test_label = np.concatenate((heartattacktestlabel, noattacktestlabel))
np.save("train_data", train)
np.save("train_label", train_label)
np.save("test_data", test)
np.save("test_label", test_label)
