import pandas as pd
import openpyxl
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def load_excel(path):
    df = pd.read_excel(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].apply(lambda x: 1 if x == 1 else 0).values
    return X, y

def manage(X, spare):
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

def interpol(data, origin, new):
    x = np.linspace(0, origin - 1, origin)
    y = data
    plt.plot(x, y, "ro")
    
    x_new = np.linspace(0, origin - 1, new)
    for kind in ["quadratic", "cubic", "zero", "slinear"]:
        f = interpolate.interp1d(x, y, kind=kind)
        y_new = f(x_new)
        plt.plot(x_new, y_new, label=str(kind))
    
    plt.legend(loc="lower right")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Interpolation Methods')
    plt.show()
    return y_new

path = 'C:/Users/91702/OneDrive/Desktop/tmp/EMG_CNN-main/ANN_SensorDataSet/bu_data_for_ML.xlsx'
X, y = load_excel(path)
print(y)

test_y = X[0]
test_x = np.linspace(0, len(test_y) - 1, len(test_y))

xnew = np.linspace(0, len(test_y) - 1, 1000)
for kind in ["quadratic", "cubic", "zero", "slinear"]:
    f = interpolate.interp1d(test_x, test_y, kind=kind)
    ynew = f(xnew)
    plt.plot(xnew, ynew, label=str(kind))

plt.plot(test_x, test_y, "ro")
plt.legend(loc="lower right")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Interpolation Methods')
plt.show()
