import pandas as pd
import numpy as np

from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP

from Classes.data_processor_class import data_process

def polynomial_main(model_structure: dict, include_bias: bool = False, 
                    include_plots: bool = False, save_results: bool = True):
    
    """
    polynomial_main: function that controls the training, fiting and validation
    of a polynomial regression model

    Inputs:
    - model_structure: the primary inpputs for the model. Contains the most 
    important parameters such as: polynomial degree and train dataset size, dictionary
    - include_bias: directly linked to the "include_bias" input of the 
    PolynomialFeatures method of sklearn, boolean
    - include_plots: boolean variable to check whether to include the learning curves
    and other plots generated to the final result. If yes, plots are shown and saved, 
    if not then nothing happens, boolean
    - save_results: boolean variable the check whether to save the results of the model 
    or not, boolean

    Outputs:
    - 
    """
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
    op = "Idle"
    print()
    print(f"Now executing: {op}")
    
    ## Unpack variables
    train_split = model_structure[op]["Train split"]
    include_development = model_structure[op]["Include development split"] 
    parameters = {"Degrees": model_structure[op]["Degree of polynomial"], 
                  "Include Bias": include_bias}
    include_learning_curve = include_plots

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
        train_split = train_split, 
        include_dev = include_development
    )

    # Train models class and Polynomial Regressor. Scaler included in the model
    polynomialReg = models_per_OP(X_train = X_train, X_test = X_test,
                                y_train = y_train, y_test = y_test)

    # Create report, save directory and return paths
    primary_inputs = pd.DataFrame(data = model_structure)

    secondary_inputs_dict = {
        "Include Bias": include_bias,
        "Include Plots": include_plots, 
        "Save results": save_results
    }
    secondary_inputs = pd.DataFrame(data = secondary_inputs_dict, index = ["Value"])

    if save_results == True:
        error_save_path, plots_save_path = data_process.data_saver(
            input_params = primary_inputs, secondary_inputs = secondary_inputs, 
            model = "Polynomial Regression")
    else: 
        error_save_path, plots_save_path = None, None

    # Initialize model
    model, modelFeatures, scaler, train_results, test_results = polynomialReg.polReg(
        parameters = parameters)

    # Get metrics
    metrics = polynomialReg.performance_metrics(train = train_results, test = test_results, error_save_path = error_save_path, operating_point = op)
    print(f"Polynomial Regression, Operating point: {op} metrics")
    print(metrics.head())

    # Learning curve
    if include_learning_curve == True:
        polynomialReg.Learning_curve(data = df_final_idle, scaler = scaler, model = model, model_features = modelFeatures, 
                        operating_point = op, plots_save_path = plots_save_path)
    else: pass


    ## Take-off
    op = "T/O"
    
    ## Unpack variables
    train_split = model_structure[op]["Train split"]
    include_development = model_structure[op]["Include development split"] 
    parameters = {"Degrees": model_structure[op]["Degree of polynomial"], 
                  "Include Bias": include_bias}

    # Get data
    df_final_to = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)

    # Features and response
    features = df_final_to.filter(["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {op} (kg/sec)"])
    response = df_final_to[f"NOx EI {op} (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = train_split, 
        include_dev = include_development
    )

    # Train models class and Polynomial Regressor. Scaler included in the model
    polynomialReg = models_per_OP(X_train = X_train, X_test = X_test,
                                y_train = y_train, y_test = y_test)

    # Initialize model
    model, modelFeatures, scaler, train_results, test_results = polynomialReg.polReg(
        parameters = parameters 
    )

    # Get metrics
    metrics = polynomialReg.performance_metrics(train = train_results, test = test_results,
                                                error_save_path = error_save_path, operating_point = op)
    print(f"Polynomial Regression, Operating point: {op} metrics")
    print(metrics.head())

    # Learning curve
    if include_learning_curve == True:
        polynomialReg.Learning_curve(data = df_final_to, scaler = scaler, model = model, model_features = modelFeatures, 
                        operating_point = op, plots_save_path = plots_save_path)
    else: pass


    ## Climb-out
    op = "C/O"
    
    ## Unpack variables
    train_split = model_structure[op]["Train split"]
    include_development = model_structure[op]["Include development split"] 
    parameters = {"Degrees": model_structure[op]["Degree of polynomial"], 
                  "Include Bias": include_bias}

    # Get data
    df_final_co = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)
    df_final_co["Rated Thrust (kN)"] = df_final_co["Rated Thrust (kN)"].values.astype(float)*0.85

    # Features and response
    features = df_final_co.filter(["Pressure Ratio", f"Rated Thrust (kN)", "Fuel Flow {op} (kg/sec)"])
    response = df_final_co[f"NOx EI {op} (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = train_split, 
        include_dev = include_development
    )

    # Train models class and Polynomial Regressor. Scaler included in the model
    polynomialReg = models_per_OP(X_train = X_train, X_test = X_test,
                                y_train = y_train, y_test = y_test)

    model, modelFeatures, scaler, train_results, test_results = polynomialReg.polReg(
        parameters = parameters)
    
    # Get metrics
    metrics = polynomialReg.performance_metrics(train = train_results, test = test_results,
                                                error_save_path = error_save_path, operating_point = op)
    print(f"Polynomial Regression, Operating point: {op} metrics")
    print(metrics.head())

    # Learning curve
    if include_learning_curve == True:
        polynomialReg.Learning_curve(data = df_final_co, scaler = scaler, model = model, model_features = modelFeatures, 
                        operating_point = op, plots_save_path = plots_save_path)
    else: pass


    ## Approach
    op = "App"
    
    ## Unpack variables
    train_split = model_structure[op]["Train split"]
    include_development = model_structure[op]["Include development split"] 
    parameters = {"Degrees": model_structure[op]["Degree of polynomial"], 
                  "Include Bias": include_bias}

    # Get data
    df_final_app = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)
    df_final_app["Rated Thrust (kN)"] = 0.3*df_final_app["Rated Thrust (kN)"].values.astype(float)

    # Features and response
    features = df_final_app.filter(["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {op} (kg/sec)"])
    response = df_final_app[f"NOx EI {op} (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = train_split, 
        include_dev = include_development
    )

    # Train models class and Polynomial Regressor. Scaler included in the model
    polynomialReg = models_per_OP(X_train = X_train, X_test = X_test,
                                y_train = y_train, y_test = y_test)

    model, modelFeatures, scaler, train_results, test_results = polynomialReg.polReg(
        parameters = parameters 
    )

    # Get metrics
    metrics = polynomialReg.performance_metrics(train = train_results, test = test_results,
                                                error_save_path = error_save_path, operating_point = op)
    print(f"Polynomial Regression, Operating point: {op} metrics")
    print(metrics.head())

    # Learning curve
    if include_learning_curve == True:
        polynomialReg.Learning_curve(data = df_final_app, scaler = scaler, model = model, model_features = modelFeatures, 
                        operating_point = op, plots_save_path = plots_save_path)
    else: pass

    ## Save results ##

