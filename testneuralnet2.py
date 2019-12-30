from sklearn.neural_network import MLPClassifier
import numpy as np

data = np.load("normalized_train_data.npy")
target = np.load("train_label.npy")

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2),
                    random_state=1)

clf.fit(data, target)

test_data = np.load("normalized_test_data.npy")
test_target = np.load("test_label.npy")
print(clf.predict_proba(test_data))
print(clf.predict(test_data))

print(test_target)

matches = 0
for i in range(len(test_target)):
    if test_target[i] == clf.predict(test_data)[i]:
        matches += 1

print(matches/len(test_target))
