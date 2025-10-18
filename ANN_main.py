import pandas as pd
from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP
from typing import Optional

def ann_main(model_structure: dict, device: str, include_plots: Optional[bool] = False, save_results: Optional[bool] = True):
    """
    ann_main:

    Inputs:
    - model_structure: the structure of the ann structured per operating point like this:
    {<Operating point>: {<Parameter name>: <Parameter value>}}, dictionary
    - device: the device to be used for execution for all operating points, str
    - include_plots: boolean statement to include or not include the loss plots,
    - save_results: boolean statement to save or not save the results

    Outputs:
    -  
    
    """
    ## Import data ##
    df_og = pd.read_csv(r"Databank/ICAO_data.csv", delimiter = ";")
    clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
            "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
            "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
            "NOx EI App (g/kg)"]
    drange = [[61, 169]]
    df_cleaned = data_process.csv_cleanup(df = df_og, clmns = clmns, drange = drange, reset_index = True, save_to_csv = True, path = "Databank/CFM56data.csv")

    ## Idle ##
    # Ready data
    op = "Idle"

    # Unpack data from model_structure dictionary
    training_split = model_structure[op]["Training split"]
    include_development = model_structure[op]["Include development split"]
    epochs = model_structure[op]["Epochs"]
    optimizer = model_structure[op]["Optimizer"]
    learning_rate = model_structure[op]["Learning rate"]
    activation = model_structure[op]["Activation Function"]
    fclayers = model_structure[op]["Number of FC layers"]
    num_nodes = model_structure[op]["Number of nodes per layer"]

    print()
    print(f"Now executing: {op}")

    df_final_idle = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "Idle")
    df_final_idle["Rated Thrust (kN)"] = df_final_idle["Rated Thrust (kN)"].values.astype(float)*0.07

    # Features and response
    features = df_final_idle.filter(["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)"])
    response = df_final_idle["NOx EI Idle (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = training_split, 
        include_dev = include_development
    )

    # Data from dataframes to custom datasets
    train_data = pd.concat([X_train, y_train], axis = 1)

    test_data = pd.concat([X_test, y_test], axis = 1)

    # Save input parameters 
    input_params = pd.DataFrame(
        data = model_structure
    )
    secondary_inputs = pd.DataFrame(
        data = {
          "Execution device": device, 
          "Include plots": include_plots,
          "Save results": save_results  
        },
        index = ["Value"]
    )

    # Create report, save directory and return paths
    if save_results == True:
        error_save_path, plots_save_path  = data_process.data_saver(input_params, secondary_inputs, model = "ANN")
    else:
        error_save_path = None
        plots_save_path = None

    # Initialize model
    models_per_OP.ann.ann_creation(operating_point = op, train_data=train_data, test_data=test_data, 
                                epochs = epochs, learning_rate = learning_rate, 
                                optimizer_sel = optimizer, activation_f = activation,
                                num_fc_layers = fclayers, num_nodes_per_layer = num_nodes , 
                                device = device, include_plots = include_plots, 
                                error_save_path = error_save_path, plots_save_path = plots_save_path)

    ## T/O ##
    # Ready data
    op = "T/O"

    # Unpack data from model_structure dictionary
    training_split = model_structure[op]["Training split"]
    include_development = model_structure[op]["Include development split"]
    epochs = model_structure[op]["Epochs"]
    optimizer = model_structure[op]["Optimizer"]
    learning_rate = model_structure[op]["Learning rate"]
    activation = model_structure[op]["Activation Function"]
    fclayers = model_structure[op]["Number of FC layers"]
    num_nodes = model_structure[op]["Number of nodes per layer"]

    print()
    print(f"Now executing: {op}")

    df_final_to = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)
    df_final_to["Rated Thrust (kN)"] = df_final_to["Rated Thrust (kN)"].values.astype(float)*0.07

    # Features and response
    features = df_final_to.filter(["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {op} (kg/sec)"])
    response = df_final_to[f"NOx EI {op} (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = training_split, 
        include_dev = False
    )

    # Data from dataframes to custom datasets
    train_data = pd.concat([X_train, y_train], axis = 1)

    test_data = pd.concat([X_test, y_test], axis = 1)

   # Initialize model
    models_per_OP.ann.ann_creation(operating_point = op, train_data=train_data, test_data=test_data, 
                                epochs = epochs, learning_rate = learning_rate, 
                                optimizer_sel = optimizer, activation_f = activation,
                                num_fc_layers = fclayers, num_nodes_per_layer = num_nodes , 
                                device = device, include_plots = include_plots,
                                error_save_path = error_save_path, plots_save_path = plots_save_path)

    ## C/O ##
    # Ready data
    op = "C/O"

    # Unpack data from model_structure dictionary
    training_split = model_structure[op]["Training split"]
    include_development = model_structure[op]["Include development split"]
    epochs = model_structure[op]["Epochs"]
    optimizer = model_structure[op]["Optimizer"]
    learning_rate = model_structure[op]["Learning rate"]
    activation = model_structure[op]["Activation Function"]
    fclayers = model_structure[op]["Number of FC layers"]
    num_nodes = model_structure[op]["Number of nodes per layer"]

    print()
    print(f"Now executing: {op}")

    df_final_co = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)
    df_final_co["Rated Thrust (kN)"] = df_final_co["Rated Thrust (kN)"].values.astype(float)*0.07

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

    # Data from dataframes to custom datasets
    train_data = pd.concat([X_train, y_train], axis = 1)

    test_data = pd.concat([X_test, y_test], axis = 1)

    # Initialize model
    models_per_OP.ann.ann_creation(operating_point = op, train_data=train_data, test_data=test_data, 
                                epochs = epochs, learning_rate = learning_rate, 
                                optimizer_sel = optimizer, activation_f = activation,
                                num_fc_layers = fclayers, num_nodes_per_layer = num_nodes , 
                                device = device, include_plots = include_plots,
                                error_save_path = error_save_path, plots_save_path = plots_save_path)

    ## App ##
    # Ready data
    op = "App"

    # Unpack data from model_structure dictionary
    training_split = model_structure[op]["Training split"]
    include_development = model_structure[op]["Include development split"]
    epochs = model_structure[op]["Epochs"]
    optimizer = model_structure[op]["Optimizer"]
    learning_rate = model_structure[op]["Learning rate"]
    activation = model_structure[op]["Activation Function"]
    fclayers = model_structure[op]["Number of FC layers"]
    num_nodes = model_structure[op]["Number of nodes per layer"]


    print()
    print(f"Now executing: {op}")

    df_final_app = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = op)
    df_final_app["Rated Thrust (kN)"] = df_final_app["Rated Thrust (kN)"].values.astype(float)*0.07

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

    # Data from dataframes to custom datasets
    train_data = pd.concat([X_train, y_train], axis = 1)

    test_data = pd.concat([X_test, y_test], axis = 1)

    # Initialize model
    models_per_OP.ann.ann_creation(operating_point = op, train_data=train_data, test_data=test_data, 
                                epochs = epochs, learning_rate = learning_rate, 
                                optimizer_sel = optimizer, activation_f = activation,
                                num_fc_layers = fclayers, num_nodes_per_layer = num_nodes , 
                                device = device, include_plots = include_plots,
                                error_save_path = error_save_path, plots_save_path = plots_save_path)
