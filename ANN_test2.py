import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP
from Classes.data_plotting_class import data_plotting

# Set device to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Manual seed for iterable results
torch.manual_seed(41)

# Import data
df_og = pd.read_csv(r"Databank/ICAO_data.csv", delimiter = ";")
clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
         "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
         "NOx EI App (g/kg)"]
drange = [[61, 169]]
df_cleaned = data_process.csv_cleanup(df = df_og, clmns = clmns, drange = drange, reset_index = True, save_to_csv = True, path = "Databank/CFM56data.csv")

# Ready data - Idle
df_final_idle = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "Idle")
df_final_idle["Rated Thrust (kN)"] = df_final_idle["Rated Thrust (kN)"].values.astype(float)*0.07

# Features and response
features = df_final_idle.filter(["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)"])
response = df_final_idle["NOx EI Idle (g/kg)"]

# Split the data
X_train, y_train, X_test, y_test = data_process.splitter(
    x = features,
    y = response,
    train_split = 0.5, 
    include_dev = False
)

# Data from dataframes to custom datasets
train_data = pd.concat([X_train, y_train], axis = 1)
train_dataset = models_per_OP.ann.CustomDataset(data = train_data)

test_data = pd.concat([X_test, y_test], axis = 1)
test_dataset = models_per_OP.ann.CustomDataset(data = test_data)

# Pass data to dataloader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 30, shuffle = True, pin_memory = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 30, shuffle = True, pin_memory = True)

## ANN ##
# Instantiate ANN model
model = models_per_OP.ann.Model()
model = model.to(device)

# Define NN parameters
epochs = 200
learning_rate = 1e-3
criterion = nn.MSELoss() # Loss criterion
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
rmse_train = []
rmse_valid = []
mape_train = []
mape_valid = []

# Iterate through epochs to train
for i in range(epochs):
    
    ## Model training ##
    model.train()
    avg_rmse, avg_mape = models_per_OP.ann.train_one_epoch(model = model, optimizer = optimizer, criterion = criterion, train_loader = train_loader, device = device)
    rmse_train.append(avg_rmse)
    mape_train.append(avg_mape)

    ## Model validation ##
    model.eval()
    avg_rmse_v, avg_mape_v = models_per_OP.ann.validate_one_epoch(model = model, criterion=criterion, test_loader=test_loader, device=device)
    rmse_valid.append(avg_rmse_v)
    mape_valid.append(avg_mape_v)
    
    # Print results    
    if i % (epochs // 5) == 0:
        print()
        print(f"Epoch \t Avg Training RMSE \t Avg Validation RMSE \t Avg Training MAPE \t Avg Validation MAPE")
        print(f"{i} \t {avg_rmse} \t {avg_rmse_v} \t {avg_mape} \t {avg_mape_v}")

include_plots = True
if include_plots == True:
    # Plot errors
    data_plotting.ann_loss_plot(rmse_train, rmse_valid, mape_train, mape_valid, epochs)
else:
    pass


