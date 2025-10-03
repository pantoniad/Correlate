import pandas as pd
import numpy as np
from typing import Optional

from sklearn.model_selection import train_test_split

import warnings
import textwrap
import os 
import datetime

class data_process:

    def __init__(self):

        pass
        
    def csv_cleanup(df: pd.DataFrame, drange: pd.DataFrame, clmns: list, reset_index: bool = True, save_to_csv: bool = False, path: Optional[str] = None):

        """
        csv_cleanup: Used to clean up a specific data range that originated from
        the ICAO Emissions databank. 
        Main capabilities: 
        1. Drop columns, 
        2. Reset indices of the data
        3. Save to output csv file based on the path specified

        Inputs:
        - self,
        - reset_index: True if user wants  to reset index, False otherwise
        - save_to_csv: True if user wants to save to csv, False otherwise. 
                        If true, path should be given
        - path: the path that the outputed csv file will be saved

        Outputs:
        -

        """
        # Clean-up dataframe
        df_inter = df[clmns]
        df_cleaned = df_inter.iloc[range(drange[0][0], drange[0][1])]
        
        # Resent index
        if reset_index == True:
            df_cleaned = df_cleaned.reset_index()
            df_cleaned = df_cleaned.drop(["index"],  axis = 1)
        
        # Save dataframe
        if save_to_csv == True and path == None:
            raise ValueError("'path' variable not defined")
        elif save_to_csv == True and path != None:
            df_cleaned.to_csv(path)

        return df_cleaned

    def splitter(x: pd.DataFrame, y: pd.DataFrame, train_split: Optional[float] = 0.6, include_dev: bool = False):

        """
        splitter: Splits the data into train, dev and test. Only the train_split size is required
        The dev and test set sizes are similar in size and equal to 1 - trainsize*0.5.

        Inputs:
        - self
        - x: the features to be splitted, dataframe,
        - y: the response to be splitted, dataframe
        - train_split: the percentage of data used for model training, float,
        - include_dev: whether to include a development set or not, boolean

        Outputs:
        - xtrain, ytrain: data part for the training
        - xdev, ydev: data part for the development
        - xtest, ytest: data part for the testing

        """

        if include_dev == True:
           
            xtrain, Xtemp, ytrain, Ytemp = train_test_split(
                x, y, train_size=train_split, random_state=10
            ) 

            xdev, xtest, ydev, ytest = train_test_split(
                Xtemp, Ytemp, train_size=0.5, random_state=10
            )
            
            return xtrain, ytrain, xdev, ydev, xtest, ytest
        
        else:

            xtrain, xtest, ytrain, ytest = train_test_split(
                x, y, train_size=train_split, random_state=10
            )

            return xtrain, ytrain, xtest, ytest


    def df_former(df: pd.DataFrame, clmns: Optional[list] = None, rows: Optional[np.array] = np.empty([0]), parameter: Optional[str] = None):
        """
        df_former: forms the dataframe to be used based on the 
        column names and row numbering

        Inputs:
        - df: dataframe to be choped up,
        - clmns: the clmns to keep from the initial dataframe, list,
        - rows: the rows to keep from the initial dataframe, two element array:
            [start, finish],
        - parameter: a parameter that is used to identify what columns to keep, str

        Outputs:
        - df_formed: formed dataframe based on values 
        
        """

        if clmns == None and len(rows) == 0 and parameter == None:
            warnings.warn("No values passed on 'clmns', 'rows' and 'parameter'. Returning initial dataframe.")
            df_final = df
        elif len(rows) == 0: # No rows defined
            if parameter == None:
                df_final = df.filter(clmns)
            else:
                # Keep the columns based on the parameter value
                df_inter = df.filter(df.columns[df.columns.str.contains(parameter)], axis = 1)
                df_final = pd.concat([df.filter(clmns), df_inter], axis = 1)
        elif clmns == None: # No columns defined
            if parameter == None:
                df_final = df.iloc[range(rows[0], rows[1])]
            else: # If parameter is defined -> clmns are also defined
                df_inter = df.filter(df.columns[df.columns.str.contains(parameter)], axis  = 1)
                df_final = df_inter.iloc[range(rows[0], rows[1])]
        
        return df_final
    
    @staticmethod
    def data_saver(input_params: pd.DataFrame, secondary_inputs: pd.DataFrame, model: str, 
                   current_dictory: str = r"E:\Correlate\model_outputs"):
        """
        data_saver:

        Inputs:
        - input_params: dataframe that contains all the values necessary for the definition
        of the model (i.e. number of layers on a nn, degree of polynomial), pd.DataFrame
        - model_parames: model parameters,
        - current_directory: directory to be used for saving the data

        Outputs:
        - saved_params: a file created that incorporates the values of the input parameters,
        the function expression and other information about the model and the execution
        - Plots: plots are saved in the folder "Plots" under the current directory in the 
        corresponding date, time and model 
        - model_errors: a file created in the current directory in the folder "Errors" that
        contains the train and test errors and any other information regarding the outputs 
        of the model

        """
        ## Create destination directories ##
        # Get current date-time
        current_dt = datetime.datetime.now().strftime("%Y-%m-%d")
        folder_name = f"Run_{current_dt}"
        
        # Check if there is a file with the same name - First path layer
        path_test = os.path.join(current_dictory, folder_name)
        if os.path.exists(path_test):
            warnings.warn("Date stamped path already exist. Using already existing directory.")
            path_first = path_test
        else:
            warnings.warn(f"Creating fodler with name: {folder_name}")
            path_first = path_test
            os.makedirs(path_first)

        # Creat fodler based on the model currently in use
        model_path = f"{model}"
        path_test_2 = os.path.join(path_first, model_path)
        if os.path.exists(path_test_2):
            warnings.warn("Model folder already exists. Using existing.")
            path_second = path_test_2
        else:
            warnings.warn(f"Model folder does not exist. Creating one named: {model}")
            path_second = path_test_2
            os.makedirs(path_second)
        
        # Check if there is a timestamped folder 
        current_t = datetime.datetime.now().strftime("%H-%M-%S")
        time = f"ExecutionTime_{current_t}"
        path_test_3 = os.path.join(path_second, time)
        if os.path.exists(path_test_3):
            warnings.warn("Timestamped folder alredy exists. Using existing path.")
            path_third = path_test_3
        else:
            warnings.warn(f"Creating new time-stamped path with name: {time}")
            path_third = path_test_3
            os.makedirs(path_third)

        # Create subfolders for each time-stamped folder
        folder_1 = "Generated plots"
        plots_save_path = os.path.join(path_third, folder_1)
        os.makedirs(plots_save_path)

        folder_2 = f"{model} results"
        error_save_path = os.path.join(path_third, folder_2)
        os.makedirs(error_save_path)

        ## Create report ##
        report_path = os.path.join(path_third, f"model_input_summary.txt")

        # Build the report text
        full_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append(f"RESULTS REPORT - {model}")
        lines.append("="*70)
        lines.append(f"Generated (EET, UTC+2): {full_time}")
        lines.append("Author: Antoniadis Panagiotis")
        lines.append("-"*70)
        lines.append("")
        lines.append("Model input parameters")
        lines.append(input_params.to_string())
        lines.append("")
        lines.append("Model secondary inputs")
        lines.append(secondary_inputs.to_string())
        lines.append("")
        lines.append("Notes:")
        notes = textwrap.fill(
            "This report contains the inputs parameters for the corresponding model trained "
            "on the ICAO data for the CFM56 engine family. "
            "Four operating points are considered along with three features "
            "(Fuel Flow, Thrust rating and Pressure ratio) and one response (EI NOx). "
            "Along with this report, the loss plots and the averaged value (per batch) "
            "of RMSE and MAPE, for the each epoch, are saved.", width=80
        )
        lines.append(notes)
        lines.append("")
        lines.append("="*70)
        lines.append("End of report.")
        lines.append("")

        # Write data
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return error_save_path, plots_save_path 
        
        

        
        
        
