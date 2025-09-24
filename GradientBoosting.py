import pandas as pd
import numpy as np

from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP
from sklearn.preprocessing import StandardScaler

## Data used on the script ##

# Load ICAO data
df_og = pd.read_csv(r"Databank/ICAO_data.csv", delimiter = ";")

clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
         "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
         "NOx EI App (g/kg)"]

drange = [[61, 169]]

# Clea-up dataframe
df_cleaned = data_process.csv_cleanup(df = df_og, clmns = clmns, drange = drange, reset_index = True, save_to_csv = True, path = "Databank/CFM56data.csv")

# Saving the model results
models_res = {
    "Polynomial Regression": {"Idle": float(), "T/O": float(), "C/O": float(), "App": float()},
    "Gradient Boosting": {"Idle": float(), "T/O": float(), "C/O": float(), "App": float()},
    "ANN": {"Idle": float(), "T/O": float(), "C/O": float(), "App": float()}
}

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
X_train, y_train, X_dev, y_dev, X_test, y_test = data_process.splitter(
    data = df_final_idle, 
    x = features,
    y = response,
    train_split = 0.5, 
    dev_split = 0.25,
    test_split = 0.25
)

# Train models class and Polynomial Regressor. Scaler included in the model
gbr = models_per_OP(X_train = X_train, X_test = X_test,
                              y_train = y_train, y_test = y_test)

parameters = {"Degrees": 2, "Include Bias": False}
model, model_features, scaler, train_results, test_results = gbr.gradientBoosting(
    #parameters = parameters 
)

# Get metrics
metrics = gbr.performance_metrics(train = train_results, test = test_results)
print("Gradient Boosting, Operating point: Idle metrics")
print(metrics.head())

# Learning curve
include_learning_curve = True

if include_learning_curve == True:
    gbr.Learning_curve(data = df_final_idle, scaler = scaler, model = model, 
                      model_features = model_features, operating_point = "Idle")
else: pass


## Take-off

# Get data
df_final_to = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "T/O")

# Features and response
features = df_final_to.filter(["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow T/O (kg/sec)"])
response = df_final_to["NOx EI T/O (g/kg)"]

# Split the data
X_train, y_train, X_dev, y_dev, X_test, y_test = data_process.splitter(
    data = df_final_to, 
    x = features,
    y = response,
    train_split = 0.5, 
    dev_split = 0.15,
    test_split = 0.15
)

# Train models class and Polynomial Regressor. Scaler included in the model
gbr = models_per_OP(X_train = X_train, X_test = X_test,
                              y_train = y_train, y_test = y_test)

parameters = {"Degrees": 2, "Include Bias": False}
model, model_features, scaler, train_results, test_results = gbr.gradientBoosting(
    parameters = parameters 
)

# Get metrics
metrics = gbr.performance_metrics(train = train_results, test = test_results)
print("Gradient Boosting, Operating point: T/O metrics")
print(metrics.head())

# Learning curve
include_learning_curve = True

if include_learning_curve == True:
    gbr.Learning_curve(data = df_final_to, scaler = scaler, model = model, 
                      model_features = model_features, operating_point = "T/O")
else: pass


## Climb-out

# Get data
df_final_co = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "C/O")
df_final_co["Rated Thrust (kN)"] = df_final_co["Rated Thrust (kN)"].values.astype(float)*0.85

# Features and response
features = df_final_co.filter(["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow C/O (kg/sec)"])
response = df_final_co["NOx EI C/O (g/kg)"]

# Split the data
X_train, y_train, X_dev, y_dev, X_test, y_test = data_process.splitter(
    data = df_final_co, 
    x = features,
    y = response,
    train_split = 0.50, 
    dev_split = 0.25,
    test_split = 0.25
)

# Train models class and Polynomial Regressor. Scaler included in the model
gbr = models_per_OP(X_train = X_train, X_test = X_test,
                              y_train = y_train, y_test = y_test)

parameters = {"Degrees": 2, "Include Bias": False}
model, model_features, scaler, train_results, test_results = gbr.gradientBoosting(
    parameters = parameters 
)

# Get metrics
metrics = gbr.performance_metrics(train = train_results, test = test_results)
print("Polynomial Regression, Operating point: C/O metrics")
print(metrics.head())

# Learning curve
include_learning_curve = True

if include_learning_curve == True:
    gbr.Learning_curve(data = df_final_co, scaler = scaler, model = model, 
                      model_features = model_features, operating_point = "C/O")
else: pass


## Approach

# Get data
df_final_app = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "App")
df_final_app["Rated Thrust (kN)"] = df_final_app["Rated Thrust (kN)"].values.astype(float)*0.3

# Features and response
features = df_final_app.filter(["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow App (kg/sec)"])
response = df_final_app["NOx EI App (g/kg)"]

# Split the data
X_train, y_train, X_dev, y_dev, X_test, y_test = data_process.splitter(
    data = df_final_app, 
    x = features,
    y = response,
    train_split = 0.5, 
    dev_split = 0.25,
    test_split = 0.25
)

# Train models class and Polynomial Regressor. Scaler included in the model
gbr = models_per_OP(X_train = X_train, X_test = X_test,
                              y_train = y_train, y_test = y_test)

parameters = {"Degrees": 2, "Include Bias": False}
model, model_features, scaler, train_results, test_results = gbr.gradientBoosting(
    parameters = parameters 
)

# Get metrics
metrics = gbr.performance_metrics(train = train_results, test = test_results)
print("Polynomial Regression, Operating point: Approach metrics")
print(metrics.head())

# Learning curve
include_learning_curve = True

if include_learning_curve == True:
    gbr.Learning_curve(data = df_final_app, scaler = scaler, model = model, 
                      model_features = model_features, operating_point = "Approach")
else: pass

## Save results ##

