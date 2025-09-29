import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from Classes.data_processor_class import data_process

from Classes.models_class import models_per_OP

from sklearn.preprocessing import StandardScaler

torch.manual_seed(41)
model = models_per_OP.Model()

df_og = pd.read_csv(r"Databank/ICAO_data.csv", delimiter = ";")

clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
         "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
         "NOx EI App (g/kg)"]

drange = [[61, 169]]

# Clea-up dataframe
df_cleaned = data_process.csv_cleanup(df = df_og, clmns = clmns, drange = drange, reset_index = True, save_to_csv = True, path = "Databank/CFM56data.csv")

## Train Polynomial Regressor per operating point ##

## Idle

# Get data
df_final_idle = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "Idle")
df_final_idle["Rated Thrust (kN)"] = df_final_idle["Rated Thrust (kN)"].values.astype(float)*0.07
#print(df_final_idle)

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

X_train_t = torch.FloatTensor(X_train.values.astype(float))
X_test_t = torch.FloatTensor(X_test.values.astype(float))

y_train_t = torch.FloatTensor(y_train.values.astype(float))
y_test_t = torch.FloatTensor(y_test.values.astype(float))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

epochs = 1000
losses = []
for i in range(epochs):
    
    # Forward with a prediction
    y_pred = model.forward(X_train_t)

    # Measure loss
    loss = criterion(y_pred, y_train_t)

    # Keep track of losses 
    losses.append(loss.detach().numpy())

    # Print results every 10 epochs
    if i % 50 == 0:
        print(f"Epoch: {i}, Loss: {loss}")

    # Backward propagation: Feed the results back to the network to 
    # optimize the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plot results
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

with torch.no_grad():
    y_eval = model.forward(X_test_t)
    loss = criterion(y_eval, y_test_t)

print(loss)

# Predict based on X_test
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test_t):
        y_val = model.forward(data)

        # Type of flower our network thinks the flower is
        print(f'Predicted value \t Test value \t')
        print(f'{i+1}.) {str(y_val)} \t {y_test_t[i]} \t {y_val.argmax().item()}')

        # Find out if resutls are correct or not
        #if y_val.argmax().item() == y_test_t[i]:
        #    correct += 1

print(f"We got {correct} correct!")

