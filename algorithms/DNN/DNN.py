import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import matthews_corrcoef
from algorithms.DNN.loadDataset import MyDataset


# Optuna example that optimizes deep neural network using PyTorch.
class DeepNN(nn.Module):
    def __init__(self, trainx, trainy, testx, testy, args) -> None:
        super(DeepNN, self).__init__()
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy

        self.args = args
        self.in_dim = args['in_dim']
        self.out_dim = args['out_dim']
        self.batch_size = args['batch_size']
        self.epochs = args['epochs']
        self.device = args['device']

        self.lr = args['lr']
        self.n_layers = args['n_layers']
        self.n_units = args['n_units']
        self.dropout = args['dropout']

        # model
        self.layers = []
        in_features = self.in_dim
        for i in range(self.n_layers):
            out_features = self.n_units
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout))
            in_features = out_features

        last_layer = nn.Linear(in_features, self.out_dim)
        self.layers.append(last_layer)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)

        return out

    def train_dl(self, model, train_loader):
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            allLoss = 0.0
            model.train()
            for X, y in train_loader:
                X = X.float().to(self.device)
                y = y.long().to(self.device)

                optimizer.zero_grad()
                y_hat = model(X)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    allLoss += float(loss.sum())

            if epoch % 10 == 0:
                print("Epoch {}: train loss {:.5f}".format(epoch, allLoss))

    def predict_dl(self, model, test_loader):
        with torch.no_grad():
            model.eval()

            preds = []
            trues = []
            for X, y in test_loader:
                X = X.float().to(self.device)
                y = y.long().to(self.device)
                y_hat = model(X)
                y_pred = np.argmax(y_hat.cpu().numpy(), axis=1)
                preds.extend(y_pred)
                trues.extend(y.cpu().numpy())

            return preds, trues

    def define_model(self, trial):
        # Optimizing the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, 5)
        layers = []

        in_features = self.in_dim
        for i in range(n_layers):
            # out_features = trial.suggest_int("n_units_l{}".format(i), 100, 1000)
            out_features = trial.suggest_int("n_units", 100, 1000)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            # p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5)
            p = trial.suggest_float("dropout", 0.1, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features

        layers.append(nn.Linear(in_features, self.out_dim))
        # layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    def objective(self, trial):
        # Generate the model
        model = self.define_model(trial).to(self.device)

        # Generate the optimizers
        # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Get the data
        train_loader = torch.utils.data.DataLoader(dataset=MyDataset(self.trainx, self.trainy),
                                                   batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=MyDataset(self.testx, self.testy),
                                                  batch_size=self.batch_size)

        # Training of the model
        for epoch in range(self.epochs):
            model.train()
            for X, y in train_loader:
                X = X.float().to(self.device)
                y = y.long().to(self.device)

                optimizer.zero_grad()
                y_hat = model(X)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()

            # Validation of the model
            predict_y, true_y = self.predict_dl(model, test_loader)
            mcc = matthews_corrcoef(true_y, predict_y)
            trial.report(mcc, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return mcc
