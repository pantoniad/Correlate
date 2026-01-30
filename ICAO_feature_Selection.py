import pandas as pd
import numpy as np
import Classes.FuelFlow_class as ffms
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
cfm56 = cfm56.drop(["index"], axis = 1)
cfm56.to_csv("Databank/CFM56data.csv")

# Operating points
ops = ["Idle", "T/O", "C/O", "App"]

## Operating point-wise feature selection
# Iterate through the operating points
for i in ops:

    # Create string
    string = f"NOx EI {i} (g/kg)"

    # Get y axis - Response
    y = cfm56[string].values.astype(float)
    
    # Features
    
    features = cfm56[["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {i} (kg/sec)"]]

    plt.rcParams.update({
            "font.size": 16,          # default text size
            "axes.titlesize": 38,     # axes titles
            "axes.labelsize": 29,     # x/y labels
            "xtick.labelsize": 25,    # tick labels
            "ytick.labelsize": 25,
            "legend.fontsize": 10,
        })
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 11))

    fig.subplots_adjust(bottom=0.15, top=0.85)
    fig.suptitle(f"EI NOx vs ICAO Features - {i} conditions", fontsize = 38)
    fig.supylabel("Emissions Index (gNOx/kgFuel)", x = 0.06, fontsize = 30)

    ######### Subplot: pressure ratio ###########
    x = features.iloc[:,0].values.astype(float)
    x = x.reshape(-1, 1)

    # Build model
    lr = LinearRegression().fit(x,y)
    y_pred = lr.predict(x)

    # Get resulst
    fit = sm.OLS(y, x).fit()
    print(f"Ops: {i}, Feature: {features.keys()[0]}, P-value: {np.format_float_scientific(fit.pvalues[0], precision = 2)}")

    # Scatter plot
    ax1.scatter(x,y, label = f"Feature: {features.keys()[0]}", color = "royalblue")
    ax1.plot(x, y_pred, color = "red")
    ax1.set_xlabel("Overall Pressure Ratio")
    ax1.grid(color = "silver", linestyle = ":")

    ########## Subplot: Rated Thrust ##########
    x = features.iloc[:,1].values.astype(float)
    x = x.reshape(-1, 1)

    # Build model
    lr = LinearRegression().fit(x,y)
    y_pred = lr.predict(x)

    # Get resulst
    fit = sm.OLS(y, x).fit()
    print(f"Ops: {i}, Feature: {features.keys()[1]}, P-value: {np.format_float_scientific(fit.pvalues[0], precision = 2)}")

    # Scatter plot
    ax2.scatter(x,y, label = f"Feature: {features.keys()[1]}", color = "orange")
    ax2.plot(x, y_pred, color = "red")
    ax2.set_xlabel("Rated Thrust (kN)")
    ax2.grid(color = "silver", linestyle = ":")

    ######## Subplot: Fuel flow #########
    x = features.iloc[:,2].values.astype(float)
    x = x.reshape(-1, 1)

    # Build model
    lr = LinearRegression().fit(x,y)
    y_pred = lr.predict(x)

    # Get resulst
    fit = sm.OLS(y, x).fit()
    print(f"Ops: {i}, Feature: {features.keys()[2]}, P-value: {np.format_float_scientific(fit.pvalues[0], precision = 2)}")

    # Scatter plot
    ax3.scatter(x,y, label = f"Feature: {features.keys()[2]}", color = "forestgreen")
    ax3.plot(x, y_pred, color = "red")
    ax3.set_xlabel("Fuel Flow (kg/s)")
    ax3.grid(color = "silver", linestyle = ":")

    # Fix layout
    fig.tight_layout()
    print()



## Feature selection using all data combined


plt.show()