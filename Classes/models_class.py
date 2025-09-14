import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from typing import Optional
from Classes.latex_class import latex as ltx

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

    def polReg(self,
               xtrain: pd.DataFrame, ytrain: pd.DataFrame, 
               xtest: pd.DataFrame, ytest: pd.DataFrame,
               parameters: dict):

        """
        polReg:

        Inputs:

        Outputs:


        """
        
        # Upack from self
        data = self.data

        # Extract parameters
        deg = parameters["Degrees"]
        bias = parameters["Include Bias"]

        
        # Polynomial object
        poly = PolynomialFeatures(degree = deg, include_bias = bias)
        x_train_poly = poly.fit_transform(xtrain)
        x_test_poly = poly.transform(xtest)
        
        # Linear regression objet
        lin = LinearRegression()
        lin.fit(x_train_poly, ytrain)

        # Predict
        y_train_pred = lin.predict(x_train_poly)
        y_test_pred = lin.predict(x_test_poly)
        
        # Create output dataframe
        d1 = {
           "Y train": ytrain,
           "Y train Pred": y_train_pred,
        }
        
        d2 = {
           "Y test": ytest,
           "Y test Pred": y_test_pred,
        }
        
        train_results = pd.DataFrame(data = d1)
        test_results = pd.DataFrame(data = d2)

        return lin, train_results, test_results
    
    def performance_metrics(self, train: pd.DataFrame, test: pd.DataFrame):
                           # to_latex: bool = False, parameters = Optional[pd.DataFrame]):
        """
        performance_metrics:

        Inputs:

        Outputs:
        
        """
        # Extract valuers from dataframe
        ytrain = train["Y train"]
        y_train_pred = train["Y train Pred"]
        ytest = test["Y test"]
        y_test_pred = test["Y test Pred"]

        # Evaluate results
        train_mse = mean_absolute_error(ytrain, y_train_pred)
        test_mse = mean_absolute_error(ytest, y_test_pred)
        train_rmse = root_mean_squared_error(ytrain, y_train_pred)
        test_rmse = root_mean_squared_error(ytest, y_test_pred)
        train_r2 = r2_score(ytrain, y_train_pred)
        test_r2 = r2_score(ytest, y_test_pred)

        # Results to dataframe
        d = {
           "MSE":{
               "Train": train_mse,
               "Test": test_mse
           },
           "RMSE":{
               "Train": train_rmse,
               "Test": test_rmse
           },
           "R2":{
               "Train": train_r2,
               "Test": test_r2
           }
        } 
        
        metrics = pd.DataFrame(data = d)

        """
        # Save to latex table
        path = parameters["Path"]
        title = parameters["Title"]
        caption = parameters["Caption"]
        label = parameters["Label"]
        headers =  metrics.keys()

        if to_latex == True:
            ltx1 = ltx(df = d, filename = path, caption = caption, label = label, header  = headers)
            ltx1.df_to_lxTable()
        else:
            pass
        """
        return metrics