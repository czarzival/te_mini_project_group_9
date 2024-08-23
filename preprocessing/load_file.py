import torch
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim

class Cnn1d(nn.Module):
    def __init__(self, num_classes=7):
        super(Cnn1d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(80, 81, kernel_size=2, stride=1, padding=1),  # Input channels: 80, Output channels: 81
            nn.BatchNorm1d(81),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))  # Output size after pooling: 81, 5
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(81, 27, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(27),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1))  # Output size: 27, 2
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(27, 9, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(9),
            nn.ReLU())
        
        self.fc = nn.Linear(9 * 2, num_classes)  # Adjust based on output size of conv3
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layer
        logit = self.fc(out)
        logit = self.activation(logit)
        return logit

# Load model
model = Cnn1d()
model.load_state_dict(torch.load('model_best'))
model.eval()

def read_data(data="kaggle_data.xlsx"):
    resArray = []
    data = xlrd.open_workbook(data)
    table = data.sheet_by_index(0)
    for i in range(table.nrows):
        line = table.row_values(i)
        resArray.append(line)
    
    X = []
    y = []
    i = 0
    while i < len(resArray):
        onedata = []
        for n in range(10):
            if i >= len(resArray):
                break
            if n == 0:
                onedata.append(list(resArray[i][:-1]))
            elif resArray[i][-1] != resArray[i-1][-1]:
                break
            else:
                onedata.append(list(resArray[i][:-1]))
            i += 1
        
        if onedata:
            onedata = np.array(onedata)
            onedata = np.transpose(onedata)  # Reshape to (C, L, N)
            X.append(onedata)
            y.append(resArray[i-1][-1])
    
    X = np.array(X)
    X = X.astype(float)
    return X, y

X, y = read_data()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Ensure correct dimensions
X_tensor = X.view(-1, 80, 10)
y = y.view(-1)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_tensor, y, test_size=0.3, random_state=420)

# Evaluate the model
test_acc = 0
test_loss = 0
criterion = nn.CrossEntropyLoss()

for i in range(len(Xtest)):
    datas = Xtest[i].unsqueeze(0)  # Add batch dimension
    if torch.cuda.is_available():
        datas = datas.cuda()
        model = model.cuda()
    
    out = model(datas)
    label = Ytest[i].unsqueeze(0)  # Add batch dimension
    
    loss = criterion(out, label)
    _, pred = torch.max(out, 1)
    
    test_acc += (pred == label).float().mean().item()
    test_loss += loss.item()

test_loss /= len(Ytest)
accu = test_acc / len(Ytest)

print(f'Test Loss: {test_loss:.3f} | Accuracy: {accu:.3f}')

