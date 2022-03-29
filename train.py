import sklearn
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def crossValidate(X, y, trainfunc, evalfunc, accfunc, folds=10):
    accuracies = []
    kf = model_selection.KFold(n_splits = folds, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = trainfunc(X_train, y_train)
        y_pred = evalfunc(model, X_test)
        accuracies.append(accfunc(y_pred, y_test))

    return np.mean(accuracies)

def mse(y_pred, y_test):
    return np.mean((y_pred - y_test)**2)

def trainOLS(X, y):
    return LinearRegression().fit(X, y)

def evalsklearn(model, X):
    return model.predict(X)

def trainRidge(X, y):
    return Ridge().fit(X, y)

def trainLasso(X, y):
    return Lasso().fit(X, y)

def trainElasticNet(X, y):
    return ElasticNet().fit(X, y)

def trainRandomForest(X, y):
    return RandomForestRegressor().fit(X, y)


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out
        
def trainMlp(X, y):
    mlp = Mlp()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.05, weight_decay=1e-5)
    criterion = nn.MSELoss()

    data = [(X[i], y[i]) for i in range(len(y))]
    trainloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=64)

    for epoch in range(4):
        running_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = torch.tensor(inputs, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.float)

            optimizer.zero_grad()
            outputs = mlp(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
            if (i+1) % 500 == 0:
              print("Samples Trained:", (i+1)*64, "running_loss:", loss.item())
              running_loss = 0

    return mlp

def evalMLP(mlp, X):
    pred = mlp(torch.tensor(X)).squeeze().detach().numpy()
    return pred


