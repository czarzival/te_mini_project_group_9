import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import torch.nn.functional as F

def load_excel(path):
    data = pd.read_excel(path, engine='openpyxl')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def manage(X, spare):
    for data in range(len(X)):
        for scalar in range(len(X[data])):
            X[data][scalar] = (X[data][scalar] // spare) * spare + spare
    return X

def expand_data(X, y, size):
    new = []
    new_label = []
    for l in range(len(X)):
        label = 0 if y[l] == 0 else 1
        for i in range(size):
            new_col = [X[l][(j - 1) * size + i] for j in range(len(X[l]) // size)]
            new_label.append(label)
            new.append(new_col)
    return np.array(new), np.array(new_label)

path = 'C:/Users/91702/OneDrive/Desktop/tmp/EMG_CNN-main/ANN_SensorDataSet/bu_data_for_ML.xlsx'
X, y = load_excel(path)

print(X.shape)
y = np.array(y)

X = manage(X, 10)
X, y = expand_data(X, y, 4)
print(X.shape)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

batch_size = 1
learning_rate = 0.0001
num_epochs = 50

model = Net(Xtrain.shape[1], 400, 200, 50, 2)

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(len(Xtrain)):
        datas = torch.tensor(Xtrain[i], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(Ytrain[i], dtype=torch.long).unsqueeze(0)

        if torch.cuda.is_available():
            datas = datas.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        out = model(datas)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(Xtrain):.4f}')

# Test
model.eval()
eval_loss = 0
eval_acc = 0
with torch.no_grad():
    for i in range(len(Xtest)):
        datas = torch.tensor(Xtest[i], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(Ytest[i], dtype=torch.long).unsqueeze(0)

        if torch.cuda.is_available():
            datas = datas.cuda()
            label = label.cuda()

        out = model(datas)
        loss = criterion(out, label)

        eval_loss += loss.item()
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).sum().item()

print(f'Test Loss: {eval_loss/len(Xtest):.6f}, Accuracy: {eval_acc/len(Xtest):.6f}')

