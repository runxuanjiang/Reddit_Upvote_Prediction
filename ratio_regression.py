import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def getDensity(ratios):
    density = KernelDensity(bandwidth=0.03, kernel='gaussian').fit(ratios.reshape(-1, 1))
    plt.hist(ratios, bins=100, density=True)
    xvals = np.linspace(0, 1, num=1000)
    yvals = np.exp(density.score_samples(xvals.reshape(-1, 1)))
    plt.plot(xvals, yvals)
    return density

class Regressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512, bias=True),
            # nn.BatchNorm1d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=512, out_features=256, bias=True),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.2, inplace=False), 
            nn.Linear(in_features=256, out_features=128, bias=True),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.2, inplace=False), 
            nn.Linear(in_features=128, out_features=64, bias=True),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.2, inplace=False), 
            nn.Linear(in_features=64, out_features=1, bias=True),
        )
    
    def forward(self, x):
        z1 = self.classifier(x)
        # z = torch.softmax(z1,dim=1)
        # zz = torch.argmax(z,dim=1)
        z = torch.sigmoid(z1).squeeze()
        # print(zz)
        return z


def custom_loss(dmm, device):
    def loss(pred, true):
        N = true.shape[0]
        error = torch.abs(pred - true)
        p = np.exp(dmm.score_samples(true.view(-1, 1).cpu()))
        return torch.mean(torch.div(error, torch.tensor(p).to(device)))

    return loss

def trainRegressor(X, y, X_test, y_test, device, density = None):
    model = Regressor(384)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=8e-4)
    criterion = nn.MSELoss()
    if density is not None:
        criterion = custom_loss(density, device)

    data = [(X[i], y[i]) for i in range(len(y))]
    trainloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=64)
    train_losses = []
    test_losses = []

    for epoch in range(100):
        model.train()
        running_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
        print("epoch: ", epoch+1, "loss: ", running_loss)
        train_losses.append(torch.sqrt(loss).item())
        with torch.no_grad():
            model.eval()
            pred = model(X_test)
            test_losses.append(torch.sqrt(criterion(pred, y_test)).item())

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.show()
    return model