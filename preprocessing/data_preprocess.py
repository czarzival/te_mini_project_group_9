import openpyxl
import numpy as np
import pandas as pd
import os

def load_excel(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    data = pd.read_excel(path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    y = np.where(y == 1, 1, 0)

    return X, y

def manage(X, spare):
    # Apply transformation
    X = np.floor_divide(X, spare) * spare
    return X.astype(int)

def expand_data(X, y, size):
    new = []
    new_label = []
    for l in range(len(X)):
        label = 1 if y[l] == 1 else 0
        for i in range(size):
            new_col = []
            for j in range(len(X[l]) // size):
                new_col.append(X[l][j * size + i])
            new_label.append(label)
            new.append(new_col)
    return np.array(new), np.array(new_label)

def write_excel(name, value):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'Sheet1'
    
    for i in range(len(value)):
        for j in range(len(value[i])):
            sheet.cell(row=j + 1, column=i + 1, value=int(value[i][j]))
    
    workbook.save(name)

path = 'C:/Users/91702/OneDrive/Desktop/tmp/EMG_CNN-main/ANN_SensorDataSet/bu_data_for_ML.xlsx'
X, y = load_excel(path)
print("Original labels:", y)

X = manage(X, 10)
new, label = expand_data(X, y, 5)
write_excel('managed_square_1_data.xlsx', X)
print("Processed data shape:", X.shape)
print("Processed data:", X)
print("Expanded data shape:", new.shape)
print("Expanded data:", new)
print("Expanded labels shape:", label.shape)
print("Expanded labels:", label)
