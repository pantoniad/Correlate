import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP

## Function definition
class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, data: pd.DataFrame):

        """
        __init__

        Inputs: 
        - data: pd.Dataframe 
        """
        feature1 = data["Pressure Ratio"]
        feature2 = data["Rated Thrust (kN)"]
        feature3 = data.loc[:, data.columns.str.contains("Fuel Flow")]
        self.features = pd.concat([feature1, feature2, feature3], axis = 1)

        response = data.loc[:, data.columns.str.contains("NOx EI")]
        self.response = response

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        features_sample = self.features.iloc[index,:].values
        response_sample = self.response.iloc[index, :].values
        return torch.tensor(features_sample.astype(np.float32())), torch.tensor(response_sample.astype(np.float32()))
    
# Set device to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
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
train_dataset = CustomDataset(data = train_data)

test_data = pd.concat([X_test, y_test], axis = 1)
test_dataset = CustomDataset(data = test_data)

# Pass data to dataloader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 30, shuffle = True, pin_memory = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 30, shuffle = True, pin_memory = True)

## ANN ##
# Instantiate ANN model
model = models_per_OP.Model()
model = model.to(device)

# Define NN parameters
epochs = 1000
learning_rate = 1e-3
criterion = nn.MSELoss() # Loss criterion
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
losses_train = []
losses_valid = []

# Iterate through epochs to train
for i in range(epochs):
    
    ## Model training ##
    model.train()
    running_loss = 0

    # Forward step - Prediction
    for j, (features_sample, response_sample) in enumerate(train_loader):
        
        # Move tensors to device
        features_sample = features_sample.to(device)
        response_sample = response_sample.to(device)

        # Train model
        optimizer.zero_grad()
        y_pred = model.forward(features_sample)
        
        # Get result from loss function
        loss = torch.sqrt(criterion(y_pred, response_sample))
        running_loss += loss

        # Update weights and optimizer
        loss.backward()
        optimizer.step()

        # Print results for each batch
        #if i % 20 == 0:
        #    print(f"Epoch \t Batch \t Loss")
        #    print(f"{i} \t {j} \t {loss}")

    avg_loss = running_loss/(j+1)
    losses_train.append(avg_loss.cpu().detach().numpy())

    ## Model validation ##
    runnign_loss = 0
    model.eval()
    with torch.no_grad():
        for j, (features_sample, response_sample) in enumerate(test_loader):
            
            # Trasfer tensors to device
            features_sample = features_sample.to(device)
            response_sample = response_sample.to(device)

            # Predict based on the trained model
            y_pred_v = model(features_sample)
            loss_v = torch.sqrt(criterion(y_pred_v, response_sample))
            running_loss += loss_v
        
        avg_loss_v = running_loss /(j+1)
        losses_valid.append(avg_loss_v.cpu().detach().numpy())
        
        if i % 200 == 0:
            print()
            print(f"Epoch \t Avg Training loss \t Avg Validation loss")
            print(f"{i} \t {avg_loss} \t {avg_loss_v}")
            
    

plt.plot(range(epochs), losses_train, label = "Train loss", color = "royalblue")
plt.plot(range(epochs), losses_valid, label = "Validation loss", color = "darkorange")
plt.xlabel("Epochs")
plt.ylabel("Root Mean Squared Error value (gNOx/kgFuel)")
plt.grid(color = "silver", linestyle = ":")
plt.title("Train and Validation error - ANN")
plt.legend()
plt.show()
