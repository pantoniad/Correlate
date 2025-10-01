import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP
from Classes.data_plotting_class import data_plotting

from typing import Optional

import warnings

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
        train_split = 0.5, 
        include_dev = False
    )

    # Data from dataframes to custom datasets
    train_data = pd.concat([X_train, y_train], axis = 1)

    test_data = pd.concat([X_test, y_test], axis = 1)

    # Unpack data from model_structure dictionary
    epochs = model_structure["Idle"]["Epochs"]
    optimizer = model_structure["Idle"]["Optimizer"]
    learning_rate = model_structure["Idle"]["Learning rate"]
    activation = model_structure["Idle"]["Activation Function"]
    fclayers = model_structure["Idle"]["Number of FC layers"]
    num_nodes = model_structure["Idle"]["Number of nodes per layer"]


    # Initialize model
    models_per_OP.ann.ann_creation(operating_point = op, train_data=train_data, test_data=test_data, 
                                epochs = epochs, learning_rate = learning_rate, 
                                optimizer_sel = optimizer, activation_f = activation,
                                num_fc_layers = fclayers, num_nodes_per_layer = num_nodes , 
                                device = device, include_plots = include_plots, save_results = save_results)

    ## T/O ##
    # Ready data
    op = "T/O"
    print()
    print(f"Now executing: {op}")

    df_final_idle = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "T/O")
    df_final_idle["Rated Thrust (kN)"] = df_final_idle["Rated Thrust (kN)"].values.astype(float)*0.07

    # Features and response
    features = df_final_idle.filter(["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow T/O (kg/sec)"])
    response = df_final_idle["NOx EI T/O (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = 0.5, 
        include_dev = False
    )

    # Data from dataframes to custom datasets
    train_data = pd.concat([X_train, y_train], axis = 1)

    test_data = pd.concat([X_test, y_test], axis = 1)

    # Unpack data from model_structure dictionary
    epochs = model_structure["T/O"]["Epochs"]
    optimizer = model_structure["T/O"]["Optimizer"]
    learning_rate = model_structure["T/O"]["Learning rate"]
    activation = model_structure["T/O"]["Activation Function"]
    fclayers = model_structure["T/O"]["Number of FC layers"]
    num_nodes = model_structure["T/O"]["Number of nodes per layer"]


    # Initialize model
    models_per_OP.ann.ann_creation(operating_point = op, train_data=train_data, test_data=test_data, 
                                epochs = epochs, learning_rate = learning_rate, 
                                optimizer_sel = optimizer, activation_f = activation,
                                num_fc_layers = fclayers, num_nodes_per_layer = num_nodes , 
                                device = device, include_plots = include_plots, save_results = save_results)

    ## C/O ##
    # Ready data
    op = "C/O"
    print()
    print(f"Now executing: {op}")

    df_final_idle = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "C/O")
    df_final_idle["Rated Thrust (kN)"] = df_final_idle["Rated Thrust (kN)"].values.astype(float)*0.07

    # Features and response
    features = df_final_idle.filter(["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow C/O (kg/sec)"])
    response = df_final_idle["NOx EI C/O (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = 0.5, 
        include_dev = False
    )

    # Data from dataframes to custom datasets
    train_data = pd.concat([X_train, y_train], axis = 1)

    test_data = pd.concat([X_test, y_test], axis = 1)

    # Unpack data from model_structure dictionary
    epochs = model_structure["C/O"]["Epochs"]
    optimizer = model_structure["C/O"]["Optimizer"]
    learning_rate = model_structure["C/O"]["Learning rate"]
    activation = model_structure["C/O"]["Activation Function"]
    fclayers = model_structure["C/O"]["Number of FC layers"]
    num_nodes = model_structure["C/O"]["Number of nodes per layer"]


    # Initialize model
    models_per_OP.ann.ann_creation(operating_point = op, train_data=train_data, test_data=test_data, 
                                epochs = epochs, learning_rate = learning_rate, 
                                optimizer_sel = optimizer, activation_f = activation,
                                num_fc_layers = fclayers, num_nodes_per_layer = num_nodes , 
                                device = device, include_plots = include_plots, save_results = save_results)

    ## App ##
    # Ready data
    op = "App"
    print()
    print(f"Now executing: {op}")

    df_final_idle = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "App")
    df_final_idle["Rated Thrust (kN)"] = df_final_idle["Rated Thrust (kN)"].values.astype(float)*0.07

    # Features and response
    features = df_final_idle.filter(["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow App (kg/sec)"])
    response = df_final_idle["NOx EI App (g/kg)"]

    # Split the data
    X_train, y_train, X_test, y_test = data_process.splitter(
        x = features,
        y = response,
        train_split = 0.5, 
        include_dev = False
    )

    # Data from dataframes to custom datasets
    train_data = pd.concat([X_train, y_train], axis = 1)

    test_data = pd.concat([X_test, y_test], axis = 1)

    # Unpack data from model_structure dictionary
    epochs = model_structure["App"]["Epochs"]
    optimizer = model_structure["App"]["Optimizer"]
    learning_rate = model_structure["App"]["Learning rate"]
    activation = model_structure["App"]["Activation Function"]
    fclayers = model_structure["App"]["Number of FC layers"]
    num_nodes = model_structure["App"]["Number of nodes per layer"]


    # Initialize model
    models_per_OP.ann.ann_creation(operating_point = op, train_data=train_data, test_data=test_data, 
                                epochs = epochs, learning_rate = learning_rate, 
                                optimizer_sel = optimizer, activation_f = activation,
                                num_fc_layers = fclayers, num_nodes_per_layer = num_nodes , 
                                device = device, include_plots = include_plots, save_results = save_results)

