import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt  
import matplotlib.gridspec as gridspec  
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error
import scipy

import warnings
from typing import Optional
import os

from Classes.latex_class import latex as lx
from Classes.FuelFlow_class import FuelFlowMethods as ffms
import Classes.correlations_class as correlate

class data_plotting:


    def __init__(self, df_all: pd.DataFrame, 
                 mean_points: pd.DataFrame,
                 engine_icao_eis: pd.DataFrame,
                 dtCorrs: pd.DataFrame,
                 exp: pd.DataFrame,
                 dtmodels: pd.DataFrame
                ):
        """
        Inputs:
       - df_all: 
       - mean_points:  mean values from the ICAO Datapoints,
                        Dataframe, First column are the names 
                        for each data position of the dot plot,
                        Second column are the values for each
                        data position,
        - engine_icao_eis: Emission Index values for the specific 
                        engine considered for the thesis
        - dtCorrs:  Data retrieved from the usage of 
                    correlation equations. Dataframe,
                    X axis contains the names of the 
                    authors of the correlations, Y 
                    axis contains the EI values for 
                    the four ICAO LTO points,
        - exp:  experimental data, Dataframe, X axis contains
                the source of the data, Y axis contains the EI
                values for the four LTO cycle points. For now
                able to integrate only one point,
        - dtmodels: 
        """
        # Required
        self.df_all = df_all
        self.mean_points = mean_points
        self.engine_icao_eis = engine_icao_eis
        self.dtCorrs = dtCorrs
        self.exp = exp
        self.dtmodels = dtmodels
        
    def distribution_plots(self,
                           method: str, 
                           size: list, 
                           title: str, 
                           xLabel: str, 
                           yLabel: str, 
                           colours: list, 
                           labels: list, 
                           dotPlotXlabel: str, 
                           dotPlotYlabel: str, 
                           lineStyle: list, 
                           Jitter :float = None,
                           save_plots_path: Optional[str] = "Empty"):
        """
        distribution_plots: a function that is able to generate four kinds of plots based on the 
                            string given in the "method" parameter of the function. 
                            Supported DISTRIBUTION plots: Dot plots, Box plots, Violin Plots, Swarm plots

        Inputs:
        - self: -
        - method:   used to determine the kind of distribution plot used
                    Inputs: "Boxplot", "Dotplot", "Swarmplot", "Violinplot"
        - size: figure size. List,
        - title: title of the plot,
        - ylimits:  the limits of the y axis of the dot plot,
                    List, [min, max, step]
        - xLabel: name of the x axis of the (whole) plot,
        - yLabel: name of the y axis of the (whole) plot
        - colours: colour palette for the dot plot. List
        - labels: Dot plot labels list,
        - Jitter: define the jitter,
        - dotPlotXlabel: the label for the x axis 
                        of the dot plot,
        - dotPlotYlabel: the label of the y axis
                        of the dot plot
        - lineStyle:  marker type for the plots of the
                        data calculated from the usage of 
                        correlation equations, list 

        Outputs:
        -   Based on the string provided for the method parameter
            a Dot, Swarm, Violin or Box plot is generated
        
        """

        # Extract data from self
        df_all = self.df_all
        mean_points = self.mean_points
        engine_icao_eis = self.engine_icao_eis
        dtCorrs = self.dtCorrs
        exp = self.exp 
        dtmodels = self.dtmodels

        # Check if all dataframes are empty 
        if df_all.empty and mean_points.empty and dtCorrs.empty and exp.empty and dtmodels.empty:
            print()
            print("No data passed. Terminating fuction")
            return
        
        # Valid distribution plot methods 
        valid_methods = ["Boxplot", "Swarmplot", "Dotplot", "Violinplot"]

        # Create palette dictionary
        paletteDict = {
            labels[0]: colours[0],
            labels[1]: colours[1],
            labels[2]: colours[2],
            labels[3]: colours[3]
        }

        # Create figure
        fig = plt.figure(figsize = (size[0], size[1]))

        # Create dot plot
        if method == "Dotplot":

            if Jitter is None:
                
                # Default jitter value
                Jitter = 0.1

                warnings.warn(
                    f"The 'Jitter' parameter must be provided. Assigning default value: {Jitter}.",
                    UserWarning
                    )
            
            #ax1 = fig.add_subplot(gs[0])
            ax = sns.stripplot(
                data = df_all,
                size = 5,
                x = dotPlotXlabel,
                hue = dotPlotXlabel,
                y = dotPlotYlabel,
                legend = False,
                palette = paletteDict
            )

            #ax.set(xticklabels = [])
            #ax.set(xlabel = None)

        # Create swarm plot
        elif method == "Swarmplot":
            ax = sns.swarmplot(
                data = df_all,
                x = dotPlotXlabel,
                y = dotPlotYlabel,
                palette = paletteDict
            )

        # Create boxplot
        elif method == "Boxplot":
            ax = sns.boxplot(
                data = df_all,
                x = dotPlotXlabel,
                y = dotPlotYlabel,
                palette = paletteDict
            )

        # Create violin plot
        elif method == "Violinplot":
            ax = sns.violinplot(
                data = df_all,
                x = dotPlotXlabel,
                y = dotPlotYlabel,
                palette = paletteDict
            )

        else:
            if not method:
                raise ValueError("No method provided. Please specify one of: "
                                + ", ".join(valid_methods))
            else:
                raise ValueError(f"Invalid method '{method}'. Choose from: "
                                + ", ".join(valid_methods))
        
        # Mean value plotting
        ax.plot(
            labels,
            mean_points.values,      
            "--*",
            markersize = 10,
            color = "black",
            zorder = 10,
            label = "Mean values - ICAO Databank"
        )

        # Specific engine EIs 
        if engine_icao_eis.empty:
            pass
        else:
            plt.plot(
                labels,
                engine_icao_eis.values[0],
                "-*",
                color = "orange",
                markersize = 10,
                label = "ICAO CFM56-7B26"
            )

        # Correlation equations value plotting
        pointer = 0
        if dtCorrs.empty:
            pass
        else:
            for i in dtCorrs.keys():
                
                # Add the data to the plot
                plt.plot(
                    labels, 
                    dtCorrs.iloc[:][i],
                    lineStyle[pointer], 
                    label = i,
                    markersize = 10
                )

                # Increase the count of the pointer
                pointer += pointer
        
        # Place the results from the models
        if dtmodels.empty:
            pass
        else:
            if dtmodels.keys()[0] == "Polynomial Regression":
                plt.plot(
                    labels,
                    dtmodels["Polynomial Regression"],
                    "-.d",
                    label = "Polynomial (2) Regression",
                    zorder = 10,
                    markersize = 10,
                    color = "greenyellow"
                )

            if dtmodels.keys()[1] == "Gradient Boosting":    
                plt.plot(
                    labels,
                    dtmodels["Gradient Boosting"],
                    "-.d",
                    label = "Gradient boosting",
                    zorder = 10,
                    markersize = 10,
                    color = "gold"
                )

            if dtmodels.keys()[2] == "ANN":
                plt.plot(
                    labels,
                    dtmodels["ANN"],
                    "-.d",
                    label = "ANN",
                    zorder = 10,
                    markersize = 10,
                    color = "deepskyblue"
                )

        # Place the experimental data
        if exp.empty:
            pass
        else:
            plt.plot(
                labels, 
                exp["Turgut - CFM56/7B26"],
                "-8",
                label = "Turgut, CFM56/7B26",
                zorder = 10,
                color = "cyan",
                markersize = 10
            )

        # Additional plot settings, Show plot
        plt.grid(color = "silver", linestyle = ":")
        plt.legend(loc = "best", fontsize = 12)
        plt.ylabel(yLabel, fontsize = 15)
        plt.yticks(fontsize = 13)
        plt.xlabel(xLabel, fontsize = 15)
        plt.xticks(fontsize = 13)
        ax.set_title(label = title, fontsize = "xx-large")
        plt.tight_layout()

        # Save plot
        if save_plots_path != "Empty":

            fig.savefig(os.path.join(save_plots_path, f"Distribution plots.png"))

        plt.show()

    def error(self):
        """
        error:  function to calculate the mean relative error between
                the correlation equations and experimental results
                with the mean points of the data for all of the operating
                points. Calculates the relative error between each
                correlation and experimental data point with the mean 
                for each operating point. Then regroups the data and 
                calculates the mean value across all operating points. 

                Can be used with any number of correlation equations and 
                experimental data sets. 

        Inputs:
        - self: - 
        
        Outputs:
        - meanRelativeEC: the mean relative error between all of the correlation
                            equations and the mean values of each operating point, 
                            for all operating points, Dataframe 
        - meanRelativeEE: same with the above, but for the experimental datasets, 
                            Dataframe
        """

        # Extract data from self
        dtCorrs = self.dtCorrs
        mean_points = self.mean_points
        exp = self.exp
        dtmodels = self.dtmodels
        
        # Constants and parameters
        operPoints = len(mean_points.index)
        corrsNum = len(dtCorrs.keys())
        modelsNum = len(dtmodels.keys())
        expNum = len(exp.keys())
        relativeEC = [] # relative error for Correlation
        relativeEE = [] # relative error for Experimental 
        relativeEM = [] # relative error for Models
        meanRelativeEC = [] # mean relative error for each correlation
        meanRelativeEE = [] # mean relative error for Experimental

        # Calculate relative error
        for i in range(0, operPoints):

            # Get parameters 
            mean = mean_points.values[i]

            # Calculate relative error - Correlation equations
            for j in range(0, corrsNum):
                
                # Calculate the relative error for each correlation j
                # at each operating point i
                error = 100*np.abs(dtCorrs.values[i][j] - mean)/mean

                # Append the value to a bigger dictionary 
                relativeEC = np.append(relativeEC, error, axis = 0)

            # Calculate relative error - Experimental data
            for j in range(0, expNum):
            
                # Calculate the relative error for each correlation j
                # at each operating point i
                error = 100*np.abs(exp.values[i][j] - mean)/mean

                # Append the value to a bigger dictionary 
                relativeEE = np.append(relativeEE, error, axis = 0)
            
            # Calculate relative error - Models
            for k in range(0, modelsNum):

                # Calculate the relative error for each model k used
                # at each operating point i
                error = 100*np.abs(dtmodels.values[i][k] - mean)/mean

                # Append the value to a bigger dictionary
                relativeEM = np.append(relativeEM, error, axis = 0)

        # Regroup the relative error lists to be 2d arrays
        chunks = [relativeEC[i:i+corrsNum] for i in range(0, len(relativeEC), corrsNum)]
        relativeECr = list(map(list, zip(*chunks)))
        
        chunks = [relativeEE[i:i+corrsNum] for i in range(0, len(relativeEE), expNum)]
        relativeEEr = list(map(list, zip(*chunks)))

        chunks = [relativeEM[i:i+modelsNum] for i in range(0, len(relativeEM), modelsNum)]
        relativeEMr = list(map(list, zip(*chunks)))

        # Get mean values
        meanRelativeEC = [np.mean(relativeECr[:][i]) for i in range(0, corrsNum)]
        meanRelativeEE = [np.mean(relativeEEr[:][i]) for i in range(0, expNum)]
        meanRelativeEM = [np.mean(relativeEMr[:][i]) for i in range(0, modelsNum)]

        # Convert into data-frames
        meanEC = pd.DataFrame(
            data = np.round(meanRelativeEC,2),
            columns = ["Mean Relative percentage error"],
            index = dtCorrs.keys()
        )

        meanEE = pd.DataFrame(
            data = np.round(meanRelativeEE,2),
            columns = ["Mean relative percentage error"],
            index = exp.keys()
        )

        meanEM = pd.DataFrame(
            data = np.round(meanRelativeEM,2),
            columns=["Mean relative percentage error"],
            index = dtmodels.keys() 

        )
        # Also convert the relative error lists
        lis = np.array(relativeECr)
        lisT = lis.T
        listT = lisT.tolist()
        relativeECd = pd.DataFrame(
            data = np.round(listT,2),
            columns = dtCorrs.keys(),
            index = dtCorrs.index
        )
        
        lis = np.array(relativeEEr)
        lisT = lis.T
        listT = lisT.tolist()
        relativeEEd = pd.DataFrame(
            data = np.round(listT,2),
            columns = exp.keys(),
            index = exp.index
        )

        lis = np.array(relativeEMr)
        lisT = lis.T
        listT = lisT.tolist()
        relativeEMd = pd.DataFrame(
            data = np.round(listT,2),
            columns = dtmodels.keys(),
            index = dtmodels.index 
        )

        # Get standard deviations
        stdEC = np.round([np.std(relativeECr[:][i]) for i in range(0, corrsNum)],2)
        stdEE = np.round([np.std(relativeEEr[:][i]) for i in range(0, expNum)],2)
        stdEM = np.round([np.std(relativeEMr[:][i]) for i in range(0, modelsNum)],2)

        # Include the standard deviation 
        meanEE["Standard Deviation"] = stdEE
        meanEC["Standard Deviation"] = stdEC
        meanEM["Standard Deviation"] = stdEM 

        return meanEC, meanEE, meanEM, relativeECd, relativeEEd, relativeEM
    
    @staticmethod
    def ann_loss_plot(rmse_train: list, rmse_valid: list, mape_train: list, mape_valid: list, 
                      epochs: int, operating_point: str, plots_save_path: str = None):
        """
        ann_loss_plot:

        Input:

        Output:
        """
     
        # Plot the losses
        fig = plt.figure(figsize=(9,7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharex = ax1)

        ax1.plot(range(epochs), rmse_train, label = "Train RMSE", color = "royalblue")
        ax1.plot(range(epochs), rmse_valid, label = "Validation RMSE", color = "darkorange")
        #ax1.set_xlabel("Number of Epochs")
        ax1.set_ylabel("RMSE (gNOx/kgFuel)")
        ax1.grid(color = "gray", linestyle = ":")
        ax1.legend()

        ax2.plot(range(epochs), [100*x for x in mape_train], label = "Train MAPE", color = "royalblue")
        ax2.plot(range(epochs), [100*x for x in mape_valid], label = "Validation MAPE", color = "darkorange")
        ax2.set_ylabel("MAPE (%)")
        ax2.set_xlabel("Number of Epochs")
        ax2.grid(color = "gray", linestyle = ":")
        ax2.legend()
        fig.suptitle(f"Train and Validation error of ANN - CFM56 family - Operating Point: {operating_point}", size = "x-large")
        fig.tight_layout()

        if plots_save_path == None:
            pass
        else:
            if operating_point == "T/O":
                operating_point = "Take-off"
                fig.savefig(os.path.join(plots_save_path, f"saved_metrics_{operating_point}.png"))
            elif operating_point == "C/O":
                operating_point = "Climb-out"
                fig.savefig(os.path.join(plots_save_path, f"saved_metrics_{operating_point}.png"))
            else:
                fig.savefig(os.path.join(plots_save_path, f"saved_metrics_{operating_point}.png"))

        plt.show()
    
    @staticmethod
    def ann_loss_plot_advanced(data_to_plot_train: pd.DataFrame, data_to_plot_test: pd.DataFrame, 
                               variable_of_interest: str, given_variable_value: int, epochs: int, 
                               sup_title: str, x_label_train: str, y_label_train: str, title_train: str,
                               x_label_test: str, y_label_test: str, title_test: str, operating_point: str,
                               plots_save_path: str = None
                               ):
        """
        ann_loss_plot_advanced: this method handles the plotting of the gridsearch conducted for the ANN. 
        It is dumped advanced because it can handle both multiple layers and multiple neurons. Its goal is
        to produce two figures, each containing a train (left) and a test (right) plot. In the first figure,
        the effect of the number of deep layers on the performance of the ANN is observed, whereas in the 
        second figure the effect of the number of neuros for a set number of layers is plotted. 
        This method does not compete with the ann_loss_plot, as the second one produces a loss plot for the given structure
        of the ANN, while this method creates a new ANN structure based on the needs of each gridsearch.

        Inputs:
        - data_to_plot_train: the data used for plotting here and are retrieved during training. Pandas dataframe with
        the following structure {}"<Variable of interest>": <value>, "Epoch": <epoch>, "Train/Test MAPE": <>, 
        "Train/Test RMSE": <>, "Train/Test R2": <>}
        - data_to_plot_test: the data used for plotting here and retrieved during testing. Pandas dataframe with a 
        structure as shown in the data_to_plot_train dataframe
        - variable_of_interest: the variable the will be used as the basis for the gridsearch. As this method is coupled 
        with the ANN, the two variables that have been tested are "No. Deep Layers" and "No. Neurons" for the given 
        structure of the data_to_plot_train/test dataframes, str
        - given_variable_value: the value given to the variable of interest by the user during initialization, int
        - epochs: the number of epochs as given by the user during initialization, int,
        - sup_title: the super-title of the plot, str,
        - x_label_train/test: the label given in the train and test x-axis of the plots of the figure, str,
        - y_label_train/test: the label given in the train and test y-axis of the plots of the figure, str
        - title_train/test: the title given in the train and test plots of the figure, str
        - operating_point: the operating point that is considered, str
        - plots_save_path: the current directory used for saving the generated plots

        Outputs:
        - Plots in the plots_save_path

        """
        # Unpack variables

        # Plot 
        variable_values = sorted(data_to_plot_train[variable_of_interest].unique())

        fig, axs = plt.subplots(1, 2, figsize = (10,6))
                
        for no_variable in variable_values:
            mape_plot_train = data_to_plot_train[data_to_plot_train[variable_of_interest] == no_variable]["Train MAPE"]
            linestyle = "-" if no_variable == given_variable_value else "--"
            axs[0].plot(
                range(0, epochs - 1, 15), 
                mape_plot_train.values[0:-1:15],
                label = f"{variable_of_interest} = {no_variable}",
                linestyle = linestyle
            )
            axs[0].set_xlabel(x_label_train)
            axs[0].set_ylabel(y_label_train)
            axs[0].grid(color="silver", linestyle=":")
            axs[0].set_title(title_train)    
            axs[0].legend()

            mape_plot_test = data_to_plot_test[data_to_plot_test[variable_of_interest] == no_variable]["Test MAPE"]
            linestyle = "-" if no_variable == given_variable_value else "--"
            axs[1].plot(range(0, epochs - 1, 15),
                        mape_plot_test.values[0:-1:15],
                        label = f"{variable_of_interest} = {no_variable}",
                        linestyle = linestyle
            )
            axs[1].set_xlabel(x_label_test)
            axs[1].set_ylabel(y_label_test)
            axs[1].grid(color="silver", linestyle=":")
            axs[1].set_title(title_test)    
            axs[1].legend()
        
        fig.suptitle(f"{sup_title} - {operating_point}") 
        fig.tight_layout()

        if plots_save_path == None:
            pass
        else:
            if operating_point == "T/O":
                operating_point = "Take-off"
                fig.savefig(os.path.join(plots_save_path, f"complexity_plots_{variable_of_interest}_ANN_{operating_point}.png"))
            elif operating_point == "C/O":
                operating_point = "Climb-out"
                fig.savefig(os.path.join(plots_save_path, f"complexity_plots_{variable_of_interest}_ANN_{operating_point}.png"))
            else:
                fig.savefig(os.path.join(plots_save_path, f"complexity_plots_{variable_of_interest}_ANN_{operating_point}.png"))

        plt.show()        

    @staticmethod
    def gbr_complexity_plot(model_params: dict,  X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_test: pd.DataFrame, op: str, model: str, plots_save_path: str = None, error_save_path: str = None):

        """    
        gbr_complexity_plot: Complexity plot generation method for the Gradient Boosting algorithm. This method 
        generates two figures that contain two plots each one for training (left) and one for testing (right). 
        The first figure depicts the effect of the number of estimators on the performance of the Decision Tree 
        base learner, and on GBR, while the second showcases the effect of the tree depth on the same paremeters

        Input:
        - model_params: dictionary that contains the parameters that define the structure of the decision tree 
        base learner. Dictionary
        - X_train/test: the features used for training/testing, pd.DataFrame,
        - y_train/test: the responses used for validating the response of the model during training/testing, 
        pd.Dataframe,
        - op: operating point considered, str,
        - model: the model used, str,
        - plots_save_path: path used for saving the generated figures, str, Defaults to None -> no saving
        
        Output:
        - Plots in the plots_save_path
        """ 
                # Unpack
        criterion = model_params[op]["Criterion"]
        learning_rate = model_params[op]["Learning rate"]
        subsample_size = model_params[op]["Subsample size"]
        n_estimators = model_params[op]["Number of estimators"]
        tree_depth = model_params[op]["Maximum Tree depth"]

        ## First case - Number of estimators vs Error + Depth
        n_estimators_min = 1
        n_estimators_max = 4*n_estimators+n_estimators_min
        n_estimators_step = int((n_estimators_max-n_estimators_min)/10) 
        
        tree_depth_min = 1
        tree_depth_max = 4*tree_depth + tree_depth_min
        tree_depth_step = int((tree_depth_max - tree_depth_min)/4)
        
        # Single parameter grid-search 
        data_to_plot_train = pd.DataFrame(columns = ["Tree depth", "No.Estimators", "Train MAPE", "Train RMSE", "Train R2"])
        data_to_plot_test = pd.DataFrame(columns = ["Tree depth", "No.Estimators", "Test MAPE", "Test RMSE", "Test R2"])

        # Gridsearch
        for i in range(tree_depth_min, tree_depth_max, tree_depth_step):
            
            if i == 1:
                depth = 1
            else:
                if i in range(2, 11):
                    depth = i
                else:
                    depth = np.round(i, -1)

            estimator = n_estimators_min
            while estimator < n_estimators_max: 

                # Extract parameters from self
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled  = scaler.transform(X_test) 

                # Initialiaze regressor
                gbr = GradientBoostingRegressor(n_estimators=estimator, learning_rate=learning_rate,
                                                criterion=criterion, max_depth=depth, subsample = subsample_size)

                # Train Regressor 
                fitted_gbr = gbr.fit(X_train_scaled, y_train)

                # Predict based on test
                y_train_pred = fitted_gbr.predict(X_train_scaled)
                y_test_pred = fitted_gbr.predict(X_test_scaled)

                # Get metrics
                train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
                test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
                train_rmse = root_mean_squared_error(y_train, y_train_pred)
                test_rmse = root_mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_data_std = np.std(y_train.values.astype(float))
                test_data_std = np.std(y_test.values.astype(float))
                train_pred_std = np.std(y_train_pred)
                test_pred_std = np.std(y_test_pred)

                # CRMSD - Train
                pred = y_train_pred
                pred_median = np.median(y_train_pred)
                real = y_train.values.astype(float) 
                real_median = np.median(y_train.values.astype(float))
                crmsd_train = np.sqrt(1/len(pred)*np.sum(((pred - pred_median)-(real - real_median))**2))

                # CRMSD - Test
                pred = y_test_pred
                pred_median = np.median(y_test_pred)
                real = y_test.values.astype(float) 
                real_median = np.median(y_test.values.astype(float))
                crmsd_test = np.sqrt(1/len(pred)*np.sum(((pred - pred_median)-(real - real_median))**2))

                # Pearson
                pearson_coeff_train = scipy.stats.pearsonr(y_train.astype(float), y_train_pred.astype(float)).statistic
                pearson_coeff_test = scipy.stats.pearsonr(y_test.astype(float), y_test_pred.astype(float)).statistic

                # Prepare data for plotting
                line_dt = {"Tree depth": depth, "No.Estimators": estimator, "Train MAPE": train_mape, "Train RMSE": train_rmse, "Train R2": train_r2,
                           "Train CRMSD": crmsd_train, "Train data based prediction": train_pred_std, "Train data Standard Deviation": train_data_std,
                           "Train Pearson Correlation coefficient": pearson_coeff_train}
                line_df = pd.DataFrame(data = line_dt, index=["Value"])
                data_to_plot_train = pd.concat([data_to_plot_train, line_df], axis = 0, ignore_index=True)

                line_dt = {"Tree depth": depth, "No.Estimators": estimator, "Test MAPE": test_mape, "Test RMSE": test_rmse, "Test R2": test_r2,
                           "Test CRMSD": crmsd_test, "Test data based prediciton": test_pred_std, "Test data Standard Deviation": test_data_std,
                           "Test Pearson Correlation coefficient": pearson_coeff_test}
                line_df = pd.DataFrame(data = line_dt, index=["Value"])
                data_to_plot_test = pd.concat([data_to_plot_test, line_df], axis = 0, ignore_index=True)

                # Increase estimator
                estimator += n_estimators_step

        # Time to plot 
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        # Get unique tree depths
        tree_depths = sorted(data_to_plot_train["Tree depth"].unique())

        # Left subplot: Train R2 vs Number of Estimators for each tree depth
        for idx, depth in enumerate(tree_depths):
            train_subset = data_to_plot_train[data_to_plot_train["Tree depth"] == depth]
            linestyle = '-' if idx == 0 else '--'
            axs[0].plot(
                train_subset["No.Estimators"],
                train_subset["Train R2"],
                label=f"Depth={depth}",
                linestyle=linestyle
            )

        axs[0].set_xlabel("Number of Estimators")
        axs[0].set_ylabel("R² Score (Train)")
        axs[0].set_title("Train R² vs Number of Estimators")
        axs[0].grid(color="silver", linestyle=":")
        axs[0].legend()

        # Right subplot: Test R2 vs Number of Estimators for each tree depth
        for idx, depth in enumerate(tree_depths):
            test_subset = data_to_plot_test[data_to_plot_test["Tree depth"] == depth]
            linestyle = '-' if idx == 0 else '--'
            axs[1].plot(
                test_subset["No.Estimators"],
                test_subset["Test R2"],
                label=f"Depth={depth}",
                linestyle=linestyle
            )

        axs[1].set_xlabel("Number of Estimators")
        axs[1].set_ylabel("R² Score (Test)")
        axs[1].set_title("Test R² vs Number of Estimators")
        axs[1].grid(color="silver", linestyle=":")
        axs[1].legend()

        fig.suptitle(f"R² vs Number of Estimators for Various Tree Depths - {op}", fontsize="x-large")
        fig.tight_layout()
        
        # Save plots and metrics
        if plots_save_path == None:
            pass
        else:
            if op == "T/O":
                op = "Take-off"
                fig.savefig(os.path.join(plots_save_path, f"complexity_plot_estimator_{op}.png"))
            elif op == "C/O":
                op = "Climb-out"
                fig.savefig(os.path.join(plots_save_path, f"complexity_plot_estimator_{op}.png"))
            else:
                fig.savefig(os.path.join(plots_save_path, f"complexity_plot_estimator_{op}.png"))
        
        if error_save_path == None:
            pass
        else:
            # Create saving paths
            complexity_save_path = os.path.join(error_save_path, f"Complexity plots results")
            if os.path.exists(complexity_save_path): 
                pass
            else: 
                os.mkdir(complexity_save_path)

            if op == "T/O":
                op = "Take-off"
                data_to_plot_train.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_train_estimator_{op}.csv"))
                data_to_plot_test.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_test_estimator_{op}.csv"))
            elif op == "C/O":
                op = "Climb-out"
                data_to_plot_train.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_train_estimator_{op}.csv"))
                data_to_plot_test.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_test_estimator_{op}.csv"))
            else:
                data_to_plot_train.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_train_estimator_{op}.csv"))
                data_to_plot_test.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_test_estimator_{op}.csv"))
       
        ## Seconde case - Tree depth vs error + Estimators
        n_estimators_min = 1
        n_estimators_max = 4*n_estimators+n_estimators_min
        n_estimators_step = int((n_estimators_max-n_estimators_min)/4) 
        
        # Struggles for low values of tree depth. If statement below fixes floating steps into 1, for low value tree depths
        tree_depth_min = 1
        tree_depth_max = 4*tree_depth + tree_depth_min
        tree_depth_step = int((tree_depth_max - tree_depth_min)/10) if (tree_depth_max - tree_depth_min)/10 > 1 else 1  
        
        # Single parameter grid-search 
        data_to_plot_train = pd.DataFrame(columns = ["Tree depth", "No.Estimators", "Train MAPE", "Train RMSE", "Train R2"])
        data_to_plot_test = pd.DataFrame(columns = ["Tree depth", "No.Estimators", "Test MAPE", "Test RMSE", "Test R2"])

        # Gridsearch
        for i in range(n_estimators_min, n_estimators_max, n_estimators_step):
            
            if i != 1:
                estimator = np.round(i, -1)
            else: 
                estimator = 1
            depth = tree_depth_min
            
            while depth < tree_depth_max: 

                # Extract parameters from self
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled  = scaler.transform(X_test) 

                # Initialiaze regressor
                gbr = GradientBoostingRegressor(n_estimators=estimator, learning_rate=learning_rate,
                                                criterion=criterion, max_depth=depth, subsample = subsample_size)

                # Train Regressor 
                fitted_gbr = gbr.fit(X_train_scaled, y_train)

                # Predict based on test
                y_train_pred = fitted_gbr.predict(X_train_scaled)
                y_test_pred = fitted_gbr.predict(X_test_scaled)

                # Get metrics
                train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
                test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
                train_rmse = root_mean_squared_error(y_train, y_train_pred)
                test_rmse = root_mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_data_std = np.std(y_train.values.astype(float))
                test_data_std = np.std(y_test.values.astype(float))
                train_pred_std = np.std(y_train_pred)
                test_pred_std = np.std(y_test_pred)

                # CRMSD - Train
                pred = y_train_pred
                pred_median = np.median(y_train_pred)
                real = y_train.values.astype(float) 
                real_median = np.median(y_train.values.astype(float))
                crmsd_train = np.sqrt(1/len(pred)*np.sum(((pred - pred_median)-(real - real_median))**2))

                # CRMSD - Test
                pred = y_test_pred
                pred_median = np.median(y_test_pred)
                real = y_test.values.astype(float) 
                real_median = np.median(y_test.values.astype(float))
                crmsd_test = np.sqrt(1/len(pred)*np.sum(((pred - pred_median)-(real - real_median))**2))

                # Pearson coefficient
                pearson_coeff_train = scipy.stats.pearsonr(y_train.astype(float), y_train_pred.astype(float)).statistic
                pearson_coeff_test = scipy.stats.pearsonr(y_test.astype(float), y_test_pred.astype(float)).statistic

                # Prepare data for plotting
                line_dt = {"Tree depth": depth, "No.Estimators": estimator, "Train MAPE": train_mape, "Train RMSE": train_rmse, "Train R2": train_r2,
                           "Train Pearson Correlation coefficient": pearson_coeff_train}
                line_df = pd.DataFrame(data = line_dt, index=["Value"])
                data_to_plot_train = pd.concat([data_to_plot_train, line_df], axis = 0, ignore_index=True)

                line_dt = {"Tree depth": depth, "No.Estimators": estimator, "Test MAPE": test_mape, "Test RMSE": test_rmse, "Test R2": test_r2,
                           "Test Pearson Correlation coefficient": pearson_coeff_test}
                line_df = pd.DataFrame(data = line_dt, index=["Value"])
                data_to_plot_test = pd.concat([data_to_plot_test, line_df], axis = 0, ignore_index=True)

               # Increase estimator
                depth += tree_depth_step

        # Get unique estimator values
        data_to_plot_test.dropna()
        data_to_plot_train.dropna()

        estimator_values = sorted(data_to_plot_train["No.Estimators"].unique())

        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        # Left subplot: Train R2 vs Tree Depth for each estimator value
        for idx, estimator in enumerate(estimator_values):
            train_subset = data_to_plot_train[data_to_plot_train["No.Estimators"] == estimator]
            linestyle = '-' if idx == 0 else '--'
            axs[0].plot(
                train_subset["Tree depth"],
                train_subset["Train R2"],
                label=f"Estimators={estimator}",
                linestyle=linestyle
            )

        axs[0].set_xlabel("Tree Depth")
        axs[0].set_ylabel("R² Score (Train)")
        axs[0].set_title("Train R² vs Tree Depth")
        axs[0].grid(color="silver", linestyle=":")
        axs[0].legend()

        # Right subplot: Test R2 vs Tree Depth for each estimator value
        for idx, estimator in enumerate(estimator_values):
            test_subset = data_to_plot_test[data_to_plot_test["No.Estimators"] == estimator]
            linestyle = '-' if idx == 0 else '--'
            axs[1].plot(
                test_subset["Tree depth"],
                test_subset["Test R2"],
                label=f"Estimators={estimator}",
                linestyle=linestyle
            )

        axs[1].set_xlabel("Tree Depth")
        axs[1].set_ylabel("R² Score (Test)")
        axs[1].set_title("Test R² vs Tree Depth")
        axs[1].grid(color="silver", linestyle=":")
        axs[1].legend()

        fig.suptitle(f"R² vs Tree Depth for Various Estimator Values - {op}", fontsize="x-large")
        fig.tight_layout()

        # Save plots and metrics
        if plots_save_path == None:
            pass
        else:
            if op == "T/O":
                op = "Take-off"
                fig.savefig(os.path.join(plots_save_path, f"complexity_plot_depth_{op}.png"))
            elif op == "C/O":
                op = "Climb-out"
                fig.savefig(os.path.join(plots_save_path, f"complexity_plot_depth_{op}.png"))
            else:
                fig.savefig(os.path.join(plots_save_path, f"complexity_plot_depth_{op}.png"))
        
        if error_save_path == None:
            pass
        else:
            # Create saving paths
            complexity_save_path = os.path.join(error_save_path, f"Complexity plots results")
            if os.path.exists(complexity_save_path): 
                pass
            else: 
                os.mkdir(complexity_save_path)

            if op == "T/O":
                op = "Take-off"
                data_to_plot_train.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_train_depth_{op}.csv"))
                data_to_plot_test.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_test_depth_{op}.csv"))
            elif op == "C/O":
                op = "Climb-out"
                data_to_plot_train.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_train_depth_{op}.csv"))
                data_to_plot_test.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_test_depth_{op}.csv"))
            else:
                data_to_plot_train.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_train_depth_{op}.csv"))
                data_to_plot_test.to_csv(os.path.join(complexity_save_path, f"complexity_metrics_test_depth_{op}.csv"))
    
        plt.show()

@staticmethod
def taylor_diagrams():

    """
    taylor_diagrams:

    Inputs:

    Outputs:
    
    """