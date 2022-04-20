
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def getCategory(ratio):
    if ratio <= 0.5:
      return 0
    elif ratio <= 0.8:
      return 1
    else:
      return 2

def getLabels(df):
    y = df['Upvote_ratio'].map(lambda x : getCategory(x)).to_numpy()
    return y


def calculateMetrics(pred, true):
    print('test accuracy:', np.mean(pred == true))
    print('f1 score:', sklearn.metrics.f1_score(pred, true, average='macro'))
    print('recall:', sklearn.metrics.recall_score(pred, true, average='macro'))
    print(sklearn.metrics.confusion_matrix(pred, true))


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 128)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(64, 3)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.drop1(out)
        out = F.relu(self.fc2(out))
        out = self.drop2(out)
        out = F.relu(self.fc3(out))
        return out
        
def trainMlp(X, y, X_test, y_test, class_weights, device):
    mlp = Mlp()
    mlp.to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0003, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    data = [(X[i], y[i]) for i in range(len(y))]
    trainloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=64)
    train_losses = []
    test_losses = []

    for epoch in range(100):
        mlp.train()
        running_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
        print("epoch: ", epoch+1, "loss: ", running_loss)
        train_losses.append(loss.item())
        with torch.no_grad():
            mlp.eval()
            pred = mlp(X_test)
            test_losses.append(criterion(pred, y_test).item())

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.show()
    return mlp

def trainGmms(X, y, X_test, y_test, num_categories, max_components):
    models = []
    for n_components in range(1, max_components + 1):
        print("Number of Components", n_components)
        gmms = []
        for i in range(num_categories):
            gmms.append(GaussianMixture(n_components).fit(X[y == i]))

        test_probs = []
        train_probs = []
        for i in range(num_categories):
            train_probs.append(gmms[i].score_samples(X))
            test_probs.append(gmms[i].score_samples(X_test))
        train_pred = np.argmax(np.array(train_probs), axis=0)
        test_pred = np.argmax(np.array(test_probs), axis=0)
        print('train accuracy:', np.mean(train_pred == y))
        calculateMetrics(test_pred, y_test)
        models.append(gmms)

    return models
          


