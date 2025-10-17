import pandas as pd
import numpy as np

from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP
from Classes.data_plotting_class import data_plotting

def gbr_main(model_structure: dict, include_learning_curve: bool = False, include_complexity_plot: bool = False, save_results: bool = True):
    
    """
    gbr_main: the main function that controls the execution of the 
    GradientBoosting algorithm.

    Inputs:
    - model_structure: a, per operating point, structured dictionary that includes
    all the primary parameters for the creation of the gradient boosting model, dictionary
    - inlcude_learning_plot: 
    - include_complexity_plot:
    - save_results: a boolean parameter that handles the saving of the results, specifically the
    numerical results. If yes, the values of the metrics are saved at a designated folder, 
    if False, nothing happens

    Outputs:

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

    ## Idle
    op = "Idle" 

    # Unpdack variables
    training_split = model_structure[op]["Train split"]
    include_development = model_structure[op]["Include development split"]
    include_learning_curve = include_learning_curve
    include_complexity_plot = include_complexity_plot
    save_results = save_results

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
        train_split = training_split, 
        include_dev = include_development
    )

    # Train models class and Polynomial Regressor. Scaler included in the model
    gbr = models_per_OP(X_train = X_train, X_test = X_test,
                                y_train = y_train, y_test = y_test)

    # Create report, save directory and return paths
    primary_inputs = pd.DataFrame(data = model_structure)

    secondary_inputs_dict = {
        "Include Learning curve": include_learning_curve, 
        "Include complexity plots": include_complexity_plot,
        "Save results": save_results
    }
    secondary_inputs = pd.DataFrame(data = secondary_inputs_dict, index = ["Value"])

    if save_results == True:
        error_save_path, plots_save_path = data_process.data_saver(
            input_params = primary_inputs, secondary_inputs = secondary_inputs, 
            model = "Gradient Boosting", notes = "Empty", gridsearch=False)
    else: 
        error_save_path, plots_save_path = None, None

    # Initialize model
    parameters = {k: v for k, v in model_structure[op].items() if k != "Train split" and k != "Include development split"}
    model, model_features, scaler, train_results, test_results = gbr.gradientBoosting(parameters)

    # Get metrics
    metrics = gbr.performance_metrics(train = train_results, test = test_results,
                                      error_save_path = error_save_path, operating_point = op)
    print()
    print(f"Gradient Boosting, Operating point: {op} metrics")
    print(metrics.head())
    print()

    # Learning curve
    if include_learning_curve == True:
        gbr.Learning_curve(data = df_final_idle, scaler = scaler, model = model, 
                    model_features = model_features, operating_point = op,
                    plots_save_path = plots_save_path)
    else: pass
    
    if include_complexity_plot == True:
        data_plotting.gbr_complexity_plot(model_params=model_structure, 
                        X_train= X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                        op = op, model = model, plots_save_path = plots_save_path)
    else: pass

    ## Take-off
    op = "T/O"
    
    # Unpack variables
    training_split = model_structure[op]["Train split"]
    include_development = model_structure[op]["Include development split"]

    # Get data
    df_final_to = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)

    # Features and response
    features = df_final_to.filter(["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {op} (kg/sec)"])
    response = df_final_to[f"NOx EI {op} (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = training_split,
        include_dev = include_development
    )

    # Train models class and Polynomial Regressor. Scaler included in the model
    gbr = models_per_OP(X_train = X_train, X_test = X_test,
                                y_train = y_train, y_test = y_test)
    
    # Initialize model
    parameters = {k: v for k, v in model_structure[op].items() if k != "Train split" and k != "Include development split"}
    model, model_features, scaler, train_results, test_results = gbr.gradientBoosting(parameters)

    # Get metrics
    metrics = gbr.performance_metrics(train = train_results, test = test_results, 
                                      operating_point = op, error_save_path = error_save_path)
    print(f"Gradient Boosting, Operating point: {op} metrics")
    print(metrics.head())
    print()

    # Learning curve
    if include_learning_curve == True:
        gbr.Learning_curve(data = df_final_to, scaler = scaler, model = model, 
                    model_features = model_features, operating_point = op,
                    plots_save_path = plots_save_path)
    else: pass
    
    if include_complexity_plot == True:
        data_plotting.gbr_complexity_plot(model_params=model_structure, 
                        X_train= X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                        op = op, model = model, plots_save_path = plots_save_path)
    else: pass

    ## Climb-out
    op = "C/O"
    
    # Unpack variables
    training_split = model_structure[op]["Train split"]
    include_development = model_structure[op]["Include development split"]

    # Get data
    df_final_co = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)
    df_final_co["Rated Thrust (kN)"] = df_final_co["Rated Thrust (kN)"].values.astype(float)*0.85

    # Features and response
    features = df_final_co.filter(["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {op} (kg/sec)"])
    response = df_final_co[f"NOx EI {op} (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = training_split, 
        include_dev = include_development 
    )

    # Train models class and Polynomial Regressor. Scaler included in the model
    gbr = models_per_OP(X_train = X_train, X_test = X_test,
                                y_train = y_train, y_test = y_test)

    # Initialize the model
    parameters = {k: v for k, v in model_structure[op].items() if k != "Train split" and k != "Include development split"}
    model, model_features, scaler, train_results, test_results = gbr.gradientBoosting(parameters)

    # Get metrics
    metrics = gbr.performance_metrics(train = train_results, test = test_results,
                                      operating_point = op, error_save_path = error_save_path)
    print(f"Gradient Boosting, Operating point: {op} metrics")
    print(metrics.head())
    print()

    # Learning curve
    if include_learning_curve == True:
        gbr.Learning_curve(data = df_final_co, scaler = scaler, model = model, 
                    model_features = model_features, operating_point = op,
                    plots_save_path = plots_save_path)
    else: pass
    
    if include_complexity_plot == True:
        data_plotting.gbr_complexity_plot(model_params=model_structure, 
                        X_train= X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                        op = op, model = model, plots_save_path = plots_save_path)
    else: pass

    ## Approach
    op = "App"
    
    # Unpack variables
    training_split = model_structure[op]["Train split"]
    include_development = model_structure[op]["Include development split"]

    # Get data
    df_final_app = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)
    df_final_app["Rated Thrust (kN)"] = df_final_app["Rated Thrust (kN)"].values.astype(float)*0.3

    # Features and response
    features = df_final_app.filter(["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {op} (kg/sec)"])
    response = df_final_app[f"NOx EI {op} (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = training_split, 
        include_dev = include_development
    )

    # Train models class and Polynomial Regressor. Scaler included in the model
    gbr = models_per_OP(X_train = X_train, X_test = X_test,
                                y_train = y_train, y_test = y_test)
    
    # Initialize the model
    parameters = {k: v for k, v in model_structure[op].items() if k != "Train split" and k != "Include development split"}
    model, model_features, scaler, train_results, test_results = gbr.gradientBoosting(parameters)

    # Get metrics
    metrics = gbr.performance_metrics(train = train_results, test = test_results,
                                      operating_point = op, error_save_path = error_save_path)
    print(f"Gradient Boosting, Operating point: {op} metrics")
    print(metrics.head())

    # Learning curve
    if include_learning_curve == True:
        gbr.Learning_curve(data = df_final_app, scaler = scaler, model = model, 
                    model_features = model_features, operating_point = op,
                    plots_save_path = plots_save_path)
    else: pass
    
    if include_complexity_plot == True:
        data_plotting.gbr_complexity_plot(model_params=model_structure, 
                        X_train= X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                        op = op, model = model, plots_save_path = plots_save_path)
    else: pass