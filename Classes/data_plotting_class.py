import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt  
import matplotlib.gridspec as gridspec  
import Classes.correlations_class as correlate
import warnings
from Classes.latex_class import latex as lx
from Classes.FuelFlow_class import FuelFlowMethods as ffms
from typing import Optional

class data_plotting:


    def __init__(self, df_all: pd.DataFrame, 
                 mean_points: pd.DataFrame,
                 dtCorrs: Optional[pd.DataFrame] = None,
                 exp: Optional[pd.DataFrame] = None,
                 dtmodels: Optional[pd.DataFrame] = None
                ):
        """
        Inputs:
       - df_all: 
       - mean_points:  mean values from the ICAO Datapoints,
                        Dataframe, First column are the names 
                        for each data position of the dot plot,
                        Second column are the values for each
                        data position
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
        
        # Optional
        if dtCorrs.empty == False:
            self.dtCorrs = dtCorrs
        
        if exp.empty == False:
            self.exp = exp

        if dtmodels.empty == False:
            self.dtmodels = dtmodels
        
    def distribution_plots(self,
                           method: str, 
                           size: list, 
                           title: str, 
                           #ylimits: list, 
                           xLabel: str, 
                           yLabel: str, 
                           colours: list, 
                           labels: list, 
                           dotPlotXlabel: str, 
                           dotPlotYlabel: str, 
                           lineStyle: list, 
                           Jitter :float = None):
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
        dtCorrs = self.dtCorrs
        exp = self.exp 
        dtmodels = self.dtmodels
        
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
        #gs = gridspec.GridSpec(3, 1, height_ratios=[9, 1, 1], figure=fig)

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
            ax1 = sns.stripplot(
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
        plt.plot(
            mean_points.index,
            mean_points.values,      
            "--*",
            markersize = 10,
            color = "black",
            zorder = 10,
            label = "Mean values - ICAO Databank"
        )

        
        # Correlation equations value plotting
        pointer = 0
        for i in dtCorrs.keys():
            
            # Add the data to the plot
            plt.plot(
                labels, 
                dtCorrs.iloc[:][i],
                lineStyle[pointer], 
                label = i 
            )

            # Increase the count of the pointer
            pointer += pointer
        
        # Place the results from the models
        plt.plot(
            labels,
            dtmodels["Polynomial Regression"],
            "-d",
            label = "Polynomial (2) Regression",
            zorder = 10
        )
        
        # Place the experimental data
        plt.plot(
            labels, 
            exp["Turgut - CFM56-7B26"],
            "-8",
            label = "Turgut, CFM56-7B26",
            zorder = 10
        )

        

        # Additional plot settings, Show plot
        plt.grid(color = "silver", linestyle = ":")
        plt.legend(loc = "best")
        plt.ylabel(yLabel)
        plt.xlabel(xLabel)
        plt.title(title)
        #minlim = 1.05*min(df_all.min(numeric_only=True).Value, dtCorrs.min().min(), exp.min().min(), dtmodels.min().min().min())
        #maxlim = 1.05*max(df_all.max(numeric_only=True).Value, dtCorrs.max().max(), exp.max().max(), dtmodels.max().max().max())
        #plt.yticks(np.arange(minlim, maxlim,100))

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
    def ann_loss_plot(rmse_train, rmse_valid, mape_train, mape_valid, epochs, ):
        """
        ann_loss_plot:

        Input:

        Output:
        """
     
        # Plot the losses
        fig = plt.figure(figsize=(9,7))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex = ax1)

        ax1.plot(range(epochs), rmse_train, label = "Train RMSE", color = "royalblue")
        ax1.plot(range(epochs), rmse_valid, label = "Validation RMSE", color = "darkorange")
        #ax1.set_xlabel("Number of Epochs")
        ax1.set_ylabel("RMSE (gNOx/kgFuel)")
        ax1.grid(color = "gray", linestyle = ":")
        ax1.legend()

        ax2.plot(range(epochs), mape_train, label = "Train MAPE", color = "royalblue")
        ax2.plot(range(epochs), mape_valid, label = "Validation MAPE", color = "darkorange")
        ax2.set_ylabel("RMSE (%)")
        ax2.set_xlabel("Number of Epochs")
        ax2.grid(color = "gray", linestyle = ":")
        ax2.legend()

        fig.suptitle("Train and Validation error - Neural Network")
        plt.show()
