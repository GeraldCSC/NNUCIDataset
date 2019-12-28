import torch
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
train_X = np.load("/home/gerald/Desktop/heartdata/NNUCIDataset/train_data.npy")
train_label = np.load("/home/gerald/Desktop/heartdata/NNUCIDataset/train_label.npy")
test_X = np.load("/home/gerald/Desktop/heartdata/NNUCIDataset/test_data.npy")
test_label = np.load("/home/gerald/Desktop/heartdata/NNUCIDataset/test_label.npy")

train_X = torch.FloatTensor(train_X)
train_label = torch.FloatTensor(train_label.reshape(train_label.shape[0],1))
test_X = torch.FloatTensor(test_X)
test_label = torch.FloatTensor(test_label.reshape(test_label.shape[0], 1))
D_in, numfirstlayer, numsecondlayer, outputdim = 12, 60, 20 , 1
dataset = TensorDataset(train_X, train_label) 
minibatch = RandomSampler(dataset)
loader = DataLoader(dataset, batch_size = 60, sampler = minibatch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class testneuralnet(torch.nn.Module):
    def __init__(self, D_in, numfirstlayer, numsecondlayer, outputdim):
        super(testneuralnet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, numfirstlayer)
        self.linear2 = torch.nn.Linear(numfirstlayer, numsecondlayer)
        self.linear3 = torch.nn.Linear(numsecondlayer, outputdim) 
        self.sigmoid = torch.nn.Sigmoid()
    "in the forward pass, we want to take in tensorinput then output the prediction"
    def forward(self,x):
        firstlayer = self.linear1(x).clamp(min=0)
        secondlayer = self.linear2(firstlayer).clamp(min=0)
        thirdlayer = self.linear3(secondlayer)
        output = self.sigmoid(thirdlayer)
        return output
#x is our prediction, y is our ground truth
def calculateaccuracy(x,y):
    x = x.clone()
    y = y.clone()
    x = x.cpu()
    y = y.cpu()
    x = x.detach().numpy()
    y = y.detach().numpy()
    x = np.where(x > 0.5 , 1, 0) 
    return accuracy_score(x,y) 

model = testneuralnet(D_in, numfirstlayer, numsecondlayer, outputdim).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.01)
for t in range(500):
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break
model.eval()
with torch.no_grad():
    test_X = test_X.to(device)
    test_pred = model(test_X)
    print("Test accuracy: " + str(calculateaccuracy(test_pred, test_label)))
