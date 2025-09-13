import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional

class models_per_OP:

    def __init__(self, data: pd.DataFrame, features: list, response: list):
        
        self.data = data
        self.features = features
        self.response = response

    def splitter(self, train_split: Optional[float] = 0.6, test_split: Optional[float] = 0.4, dev_split: Optional[float] = 0.2):

        """
        splitter

        Inputs:
        - self
        - train_split: the percentage of data used for model training, float,
        - test_split: the percentage of data used for testing, float,
        - dev_split: the percentage of data used for the development, float

        Outputs:
        - xtrain, ytrain: data part for the training
        - xdev, ydev: data part for the development
        - xtest, ytest: data part for the testing

        """
        # Extract data from self
        data = self.data
        x = self.features
        y = self.response

        # Split data: Train and temp
        xtrain, Xtemp, ytrain, Ytemp = train_test_split(
            x, y, train_size=train_split, random_state=42
        )

        # Split data: Temp to Dev and Test
        # Dev: train, Test: test, get split %
        size = dev_split/(test_split+dev_split)

        xdev, xtest, ydev, ytest = train_test_split(
            Xtemp, Ytemp, train_size=size, random_state=42
        )

        return xtrain, ytrain, xdev, ydev, xtest, ytest

    def polReg():

        """
        polReg:

        Inputs:

        Outputs:


        """
        pass
