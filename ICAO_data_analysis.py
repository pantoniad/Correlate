import pandas as pd
import numpy as np
import FuelFlow_class as ffms
import sklearn 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm    
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv(r"E:/Correlate/Databank/ICAO_data.csv", delimiter=";")

# Define and get only the collumns needed
clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
         "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
         "NOx EI App (g/kg)"]

data = df[clmns]

# Define and get only the rows for the needed engine
engRange = [[61, 169]]
cfm56 = data.iloc[range(engRange[0][0], engRange[0][1])]
cfm56 = cfm56.reset_index()

# Operating points
ops = ["Idle", "T/O", "C/O", "App"]

# Iterate through the operating points
for i in ops:

    # Create string
    string = f"NOx EI {i} (g/kg)"

    # Get y axis - Response
    y = cfm56[string].values.astype(float)
    
    # Features
    
    features = cfm56[["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {i} (kg/sec)"]]

    # Iterate through features to get p-values    
    for j in range(0, len(features.keys())):

        # Get x - Sorted
        x = features.iloc[:,j].values.astype(float)
        x = x.reshape(-1, 1)

        # Build model
        lr = LinearRegression().fit(x,y)
        y_pred = lr.predict(x)

        # Get resulst
        fit = sm.OLS(y, x).fit()
        print(f"Ops: {i}, Feature: {features.keys()[j]}, P-value: {fit.pvalues[0]}")
    
        # Scatter plot
        plt.figure(figsize = (7, 5))
        plt.scatter(x,y, label = f"Feature: {features.keys()[j]}")
        plt.plot(x, y_pred, color = "red")
        plt.ylabel("Emissions Index (g/kg)")
        plt.xlabel(features.keys()[j])
        plt.title(f"Relationship EI NOx at {i} conditions with {features.keys()[j]}")
        plt.legend()
        plt.grid(color = "silver", linestyle = ":")

    print()

plt.show()