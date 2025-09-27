import pandas as pd
import numpy as np
from typing import Optional

from sklearn.model_selection import train_test_split

import warnings

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