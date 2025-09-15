
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP

# Load data
df = pd.read_csv(r"Databank/ICAO_data.csv", delimiter = ";")

clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
         "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
         "NOx EI App (g/kg)"]

drange = [[61, 169]]

dfCleanUp = data_process(df = df, clmns = clmns, drange = drange)
df = dfCleanUp.csv_cleanup(reset_index = True, save_to_csv = True, path = "Databank/CFM56data.csv")

# Operating point splitting
dIdle = df.drop(columns= ["Fuel Flow T/O (kg/sec)", 
                          "Fuel Flow C/O (kg/sec)",
                          "Fuel Flow App (kg/sec)",
                          "NOx EI T/O (g/kg)",
                          "NOx EI C/O (g/kg)",
                          "NOx EI App (g/kg)"])

dTakeoff = df.drop(columns= ["Fuel Flow Idle (kg/sec)", 
                          "Fuel Flow C/O (kg/sec)",
                          "Fuel Flow App (kg/sec)",
                          "NOx EI Idle (g/kg)",
                          "NOx EI C/O (g/kg)",
                          "NOx EI App (g/kg)"])

dClimbout = df.drop(columns= ["Fuel Flow Idle (kg/sec)", 
                          "Fuel Flow T/O (kg/sec)",
                          "Fuel Flow App (kg/sec)",
                          "NOx EI Idle (g/kg)",
                          "NOx EI C/O (g/kg)",
                          "NOx EI App (g/kg)"])

dApp = df.drop(columns= ["Fuel Flow Idle (kg/sec)", 
                          "Fuel Flow T/O (kg/sec)",
                          "Fuel Flow C/O (kg/sec)",
                          "NOx EI Idle (g/kg)",
                          "NOx EI T/O (g/kg)",
                           "NOx EI C/O (g/kg)"])

ops = ["Idle", "T/O", "C/O", "App"]

# Iterate through the opeating points
for i in ops:
    
    print(i)
    # Keep only columns that contain the operating point 
    df1 = df.filter(df.columns[df.columns.str.contains(i)], axis=1)
    
    # Get the other columns and append
    df2 = df.filter(["Pressure Ratio", "Rated Thrust (kN)"])
    df3 = pd.concat([df2, df1], axis = 1)

    # Get features and response
    features = df3.drop(columns=f"NOx EI {i} (g/kg)")
    response = df3[f"NOx EI {i} (g/kg)"]

    # Initialize models_per_OP class
    models = models_per_OP(
        data = df3,
        features = features,
        response = response
    )

    # Split data
    X_train, y_train, X_dev, y_dev, X_test, y_test = models.splitter(
        train_split = 0.7,
        test_split = 0.15,
        dev_split = 0.15
    )

    # Train on the dev set (only applicable to Polynomial regression as of now)
    parameters = {"Degrees": 2, "Include Bias": True}
    polymodel, polyfeatures, train_poly, test_poly = models.polReg(
        xtrain = X_train, ytrain = y_train, xtest = X_dev, ytest = y_dev,
        parameters = parameters
    )
    
    # Get metrics
    metrics = models.performance_metrics(train = train_poly, test = test_poly)
    print(f"Operating point: {i} metrics")
    print(metrics.head())

    # Learning curve
    models.Learning_curve(model = polymodel, model_features = polyfeatures, 
                          operating_point = i)
    