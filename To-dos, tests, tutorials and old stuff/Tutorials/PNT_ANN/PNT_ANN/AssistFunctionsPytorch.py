import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


# Custom RMSE loss square root of MSEloss
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted, actual):
        return torch.sqrt(self.mse(predicted, actual))


# Data preparation class
class Data_prep(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    # Multilayer Perceptron class


class MLP(nn.Module):  #### change with changeable paths ####
    def __init__(self, input_size, output_size, neurons, layers, activ_func):
        super().__init__()
        layers = (layers - 1)  # pytorch sets one layer by default
        self.activation_function = activ_func
        self.input_layer = nn.Linear(input_size, neurons)
        self.middle_layers = nn.ModuleList(
            [
                nn.Linear(neurons, neurons) for i in range(layers)
            ]
        )
        self.output_layer = nn.Linear(neurons, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_function(x)
        for layer in self.middle_layers:
            x = layer(x)
            x = self.activation_function(x)

        x = self.output_layer(x)
        return x


class MLP_old(nn.Module):

    def __init__(self, input_size, output_size, neurons, ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, neurons),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.PReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.PReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.PReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            # # nn.LeakyReLU(),
            # # nn.PReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            # # nn.LeakyReLU(),
            # # nn.PReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            # # nn.LeakyReLU(),
            # # nn.PReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            # # nn.LeakyReLU(),
            # # nn.PReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            # # nn.LeakyReLU(),
            # # nn.PReLU(),
            nn.Linear(neurons, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class MLP_test(nn.Module):  #### change with changeable paths ####
    def __init__(self, input_size, output_size, neurons, layers, activ_func, variation):
        super().__init__()
        layers = (layers - 1)  # pytorch sets one layer by default
        self.activation_function = activ_func
        self.variation = variation
        self.input_layer = nn.Linear(input_size, neurons)
        self.middle_layers = nn.ModuleList(
            [
                nn.Linear(neurons, neurons) for i in range(layers)
            ]
        )
        self.output_layer = nn.Linear(neurons, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_function(x)
        for layer in self.middle_layers:
            x = layer(x)
            x = self.activation_function(x)

        x = self.output_layer(x)
        x = x + self.variation
        return x


# pre trained model loading
def load_model(model_path, input_size, output_size, neurons, layers, activ_func, mode):
    if mode == "Old":
        model = MLP_old(input_size, output_size, neurons)
    else:
        model = MLP(input_size, output_size, neurons, layers, activ_func)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model


# load the scaler
def load_scaler(scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler


# basic preprocessing
def preprocess_data(X, scaler):
    X_scaled = scaler.transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32)


# Prediction Function
def predict(model, X):
    with torch.no_grad():
        outputs = model(X).squeeze()
    return outputs.numpy()


class Sort():
    def __init__(self, engine_dataset, design_features, training_count):
        self.engine_dataset = engine_dataset
        self.design_features = design_features
        self.training_count = training_count

    def sort_data(self, weights):
        alpha = self.engine_dataset
        a = alpha.sort_values(by=['Design_id']).reset_index(drop=True)
        a1 = a.loc[a["Design_id"].isin(self.training_count)]
        a = a[self.design_features]
        a1 = a1[self.design_features]
        a = a.to_numpy()
        a1 = a1.to_numpy()
        b1 = np.repeat(a1, np.shape(a)[0], axis=0)
        b1 = np.reshape(b1, (np.shape(a1)[0], np.shape(a)[0], np.shape(a1)[1]))
        c = np.array([a])
        d1 = np.abs((c - b1) / b1)
        d2 = (c - b1) / b1
        e1 = np.sum(weights * d1, axis=2) / np.sum(weights)
        e1 = np.swapaxes(e1, 1, 0)
        e2 = np.sum(weights * d2, axis=2) / np.sum(weights)
        e2 = np.where(e2 > 0, 1, e2)
        e2 = np.where(e2 < 0, -1, e2)
        e2 = np.swapaxes(e2, 1, 0)
        f1 = np.min(e1, axis=1)
        g1 = np.repeat(f1, np.shape(a1)[0])
        g1 = np.reshape(g1, (np.shape(e1)[0], np.shape(e1)[1]))
        h1 = np.where(e1 == g1)
        diff = e2[h1]
        diff = np.where(np.isnan(diff), 0, diff)
        return f1, h1[1], diff


def total_temperature(temperature, mach, gamma):
    #  Function for calculating total temperature before PROOSIS output file
    total_temp = (1 + (((gamma - 1) / 2) * (mach ** 2))) * temperature
    return total_temp


def total_pressure(temperature, total_temperature, gamma, pressure):
    # Function for calculating total pressure before PROOSIS output file
    total_press = (total_temperature / temperature) ** (gamma / (gamma - 1)) * pressure
    return total_press
