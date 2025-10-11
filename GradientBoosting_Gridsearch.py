import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from Classes.data_processor_class import data_process

np.random.seed(34)

# Load ICAO data
df_og = pd.read_csv(r"Databank/ICAO_data.csv", delimiter = ";")

clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
        "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
        "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
        "NOx EI App (g/kg)"]

drange = [[61, 169]]

# Clea-up dataframe
df_cleaned = data_process.csv_cleanup(df = df_og, clmns = clmns, drange = drange, reset_index = True, save_to_csv = True, path = "Databank/CFM56data.csv")

## Iterate through operating points to generate grid search
ops = ["Idle","T/O", "C/O", "App"] 
primary_inputs = pd.DataFrame()

for op in ops:

    # Get data
    df_final_idle = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)
    df_final_idle["Rated Thrust (kN)"] = df_final_idle["Rated Thrust (kN)"].values.astype(float)*0.07

    # Features and response
    features = df_final_idle.filter(["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {op} (kg/sec)"])
    response = df_final_idle[f"NOx EI {op} (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = 1, 
        include_dev = False
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialiaze regressor
    gbr = GradientBoostingRegressor()

    # Set up grid search
    parameters = {
        "n_estimators": range(0, 20, 10),
        "learning_rate": np.arange(0.001, 0.03, 0.002),
        "max_depth": range(0, 20, 10),
        "subsample": np.arange(0.1, 1, 0.1)
    }
    gbr_GS = GridSearchCV(gbr, parameters, verbose = 2, cv = 7, scoring = "r2", n_jobs = -1)

    gbr_GS.fit(X_train_scaled, y_train.values)

   
    best_params = gbr_GS.best_params_
    best_score = gbr_GS.best_score_
    
    res_save = pd.DataFrame(
        data = {
            "Operating point": [op], 
            "Learning rate": best_params["learning_rate"], 
            "Maximum tree depth": best_params["max_depth"],
            "Number of estimators": best_params["n_estimators"],
            "Subsample size": best_params["subsample"],
            "Best score - R2": [gbr_GS.best_score_]
        },
        index = ["Value"]
    )
    
    primary_inputs = pd.concat([primary_inputs, res_save], axis = 0)

    print(f"Best parameters found for operating point {op}: {gbr_GS.best_params_}")
    print(f"Best cross-validation score for opeating point {op}: {gbr_GS.best_score_}")


# Save results
secondary_inputs = []
data_process.data_saver(
        input_params = primary_inputs.T, secondary_inputs = secondary_inputs, model = "Gradient Boosting - Gridsearch",
        current_dictory=r"E:\Correlate\model_outputs\Gridsearch", 
        notes = "Training on the CFM56 ICAO Emissions Databank data", gridsearch = True
)
