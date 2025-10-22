import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from Classes.data_processor_class import data_process

class CustomDataset(Dataset):

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
        return torch.tensor(features_sample.astype(float)), torch.tensor(response_sample.astype(float))
    
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

train_data = pd.concat([X_train, y_train], axis = 1)
test_data = pd.concat([X_test, y_test], axis = 1)
train_data_t = torch.FloatTensor(train_data.values.astype(float))
test_data_t = torch.FloatTensor(test_data.values.astype(float))
1
# Move data into the dataset
train_dataset = CustomDataset(data = train_data)
test_dataset = CustomDataset(data = test_data)

# Couple data to trainloader
train_loader = DataLoader(train_dataset, batch_size = 20, shuffle = True)

for i, (features_sample, response_sample) in enumerate(train_loader):
    print("Sample index:", i )
    print("Batch data", features_sample)
    print("Batch labels", response_sample)
    print("Batch size:", features_sample.shape)