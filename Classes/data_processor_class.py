import pandas as pd
import numpy as np
from typing import Optional

class data_process:

    def __init__(self, df: pd.DataFrame, drange: list, clmns: list):

        """
        Inputs: 
        - df: Dataframe from which data are parsed,
        - drange: range of rows to keep from the df,
        - clmns: columns to be kept from the df
        
        """

        self.df = df
        self.drange = drange
        self.clmns = clmns
        
    def csv_cleanup(self, reset_index: bool = True, save_to_csv: bool = False, path: Optional[str] = None):

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

        # Extract parameters from self
        df = self.df
        clmns = self.clmns
        drange = self.drange

        # Clean-up dataframe
        df2 = df[clmns]
        df2 = df2.iloc[range(drange[0][0], drange[0][1])]
        
        # Resent index
        if reset_index == True:
            df2 = df2.reset_index()
            df2 = df2.drop(["index"],  axis = 1)
        
        # Save dataframe
        if save_to_csv == True and path == None:
            raise ValueError("'path' variable not defined")
        elif save_to_csv == True and path != None:
            df2.to_csv(path)

        return df2