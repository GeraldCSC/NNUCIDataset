import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv("heart.csv") 
del data['sex']
targets = data['target'].to_numpy()
del data['target'] 
array = data.to_numpy()
X_heartattack_train, X_heartattack_test, y_heartattack_train, y_heartattack_test = train_test_split(array[:165,:], targets[:165], test_size=0.30, random_state=42)
X_noattack_train, X_noattack_test, y_noattack_train, y_noattack_test = train_test_split(array[165:,:], targets[165:], test_size=0.30, random_state=42)
