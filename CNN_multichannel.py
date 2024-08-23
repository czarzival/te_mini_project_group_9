import openpyxl
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def read_data(data=r"C:\Users\91702\OneDrive\Desktop\tmp\EMG_CNN-main\Kaggle_dataset\kaggle_data.xlsx"):
    resArray = []
    workbook = openpyxl.load_workbook(data, data_only=True)
    sheet = workbook.active
    for row in sheet.iter_rows(values_only=True):
        resArray.append(row)
    
    x = np.array(resArray)
    X = []
    y = []
    i = 0
    while i < len(resArray):
        onedata = []
        while i < len(resArray) and (i == 0 or resArray[i][-1] == resArray[i-1][-1]):
            onedata.append(list(resArray[i][:-1]))
            i += 1
            if i >= len(resArray):
                break
        if len(onedata) > 0:
            onedata = np.array(onedata).T
            X.append(onedata)
            y.append(resArray[i-1][-1])
    
    X = np.array(X).astype(float)
    y = np.array(y).astype(int)
    print(X.shape)
    print(len(y))

    return X, y

# Load and process data
X, y = read_data()

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Reshape tensors for the model
X_tensor = X.view(X.size(0), 80, 10)
y = y.view(-1)

# Split data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_tensor, y, test_size=0.3, random_state=420)

print(X_tensor.shape)  # torch.Size([630, 80, 10])
print(y.shape)        # torch.Size([630])

class Cnn1d(nn.Module):
    def __init__(self, num_class=7):
        super(Cnn1d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(80, 81, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(81),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(81, 27, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(27),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1))
        self.conv3 = nn.Sequential(
            nn.Conv1d(27, 9, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(9),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Linear(18, num_class)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(x.size(0), -1)
        logit = self.fc(out)
        logit = self.activation(logit)
        return logit

# Hyperparameters
batch_size = 2
learning_rate = 1e-4
num_epoches = 40000
valid_loss_min = float('inf')

# Initialize model, criterion, and optimizer
model = Cnn1d(num_class=7)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
loss_curve = []
tr_acc = []
for epoch in range(num_epoches):
    model.train()
    train_acc = 0
    for i in range(len(Xtrain)):
        datas = Xtrain[i].unsqueeze(0)  # Add batch dimension
        labels = Ytrain[i].unsqueeze(0)  # Add batch dimension
        if torch.cuda.is_available():
            datas = datas.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        out = model(datas)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out, 1)
        train_acc += (pred == labels).float().mean()

    acc = train_acc / len(Xtrain)
    loss_curve.append(loss.item())
    tr_acc.append(acc.item())

    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}')
        if loss.item() < valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {loss.item():.6f}). Saving model ...')
            torch.save(model.state_dict(), 'model_best.pth')
            valid_loss_min = loss.item()

# Save loss and accuracy curves
plt.plot(loss_curve)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_40000.png')
plt.show()

plt.plot(tr_acc)
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('accuracy_40000.png')
plt.show()

# Save the entire model
torch.save(model, 'whole_model.pth')

# Testing loop
model.eval()
eval_loss = 0
eval_acc = 0
with torch.no_grad():
    for i in range(len(Xtest)):
        datas = Xtest[i].unsqueeze(0)
        labels = Ytest[i].unsqueeze(0)
        if torch.cuda.is_available():
            datas = datas.cuda()
            labels = labels.cuda()

        out = model(datas)
        loss = criterion(out, labels)
        eval_loss += loss.item()

        _, pred = torch.max(out, 1)
        eval_acc += (pred == labels).float().mean()

train_loss = eval_loss / len(Xtest)
accu = eval_acc / len(Xtest)

print(f'Test Loss: {train_loss:.3f} | Accuracy: {accu:.3f}')
