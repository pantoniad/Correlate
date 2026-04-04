import pandas as pd
import numpy as np
import os
import warnings
import skill_metrics as sm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch as fancy_arrow

def add_labels(x, y1, y2, y3, w):
    for i in range(len(x)):
        plt.text(i-w, y1.iloc[i]+y1.iloc[i]*0.03, np.round(y1.iloc[i], 2), ha = "center", 
                 bbox = dict(facecolor = "silver", edgecolor = "black", alpha = 0.6), 
                 zorder = 13, fontsize = 38, fontname = "Times New Roman")
        plt.text(i, y2.iloc[i]+y2.iloc[i]*0.03, np.round(y2.iloc[i], 2), ha = "center", 
                 bbox = dict(facecolor = "silver", edgecolor = "black", alpha = 0.6),
                 zorder = 13, fontsize = 38, fontname = "Times New Roman")
        plt.text(i+w, y3.iloc[i]+y3.iloc[i]*0.03, np.round(y3.iloc[i], 2), ha = "center",
                 bbox = dict(facecolor = "silver", edgecolor = "black", alpha = 0.6),
                 zorder = 13, fontsize = 38, fontname = "Times New Roman")

# Import complexity plot data
operating_points = ["Idle", "Take-off", "Climb-out", "App"]

variable_of_interest = {"Gradient Boosting": "estimator",
                        "ANN": "LAYERS"}

surrogates_paths = {"Polynomial Regression": r"E:\Correlate\model_outputs\Run_2025-12-31\Polynomial Regression\ExecutionTime_14-58-35\Polynomial Regression results", 
                    "Gradient Boosting": r"E:\Correlate\model_outputs\Run_2025-12-31\Gradient Boosting\ExecutionTime_15-17-51\Gradient Boosting results", 
                    "ANN": r"E:\Correlate\model_outputs\Run_2025-12-31\ANN\ExecutionTime_15-42-10\ANN results"}

# Extract data: RMSE and Data Standard Deviation for each operating point and model
try:

    data_pol = pd.DataFrame()
    data_gbr = pd.DataFrame()
    data_ann = pd.DataFrame()

    for model, path in surrogates_paths.items():

        for op in operating_points:

            filename = "saved_metrics_" + op + ".csv"
            ext_path = path + "\\" + filename
            saved_metrics = pd.read_csv(ext_path)
            saved_metrics_d = saved_metrics[["RMSE", "Data Standard Deviation"]]
            saved_metrics_d = saved_metrics_d.drop([0])

            if model == "Polynomial Regression":
                data_pol = pd.concat([data_pol, saved_metrics_d], axis = 0)
            elif model == "Gradient Boosting":
                data_gbr = pd.concat([data_gbr, saved_metrics_d], axis = 0)
            elif model == "ANN":
                data_ann = pd.concat([data_ann, saved_metrics_d], axis = 0)
        
    data_pol = data_pol.reset_index(drop = "True")
    data_gbr = data_gbr.reset_index(drop = "True")
    data_ann = data_ann.reset_index(drop = "True")
            
except Exception as c:
    raise RuntimeError("Data cleaning process error:", c)

# Plot the data - Bar charts
try:

    fraction = pd.DataFrame()

    # Pre-process the data
    for i in range(len(operating_points)):

        fraction_pol = data_pol.iloc[i]["RMSE"]/data_pol.iloc[i]["Data Standard Deviation"]
        fraction_gbr = data_gbr.iloc[i]["RMSE"]/data_gbr.iloc[i]["Data Standard Deviation"]
        fraction_ann = data_ann.iloc[i]["RMSE"]/data_ann.iloc[i]["Data Standard Deviation"]
        fraction_vals = pd.DataFrame([fraction_pol, fraction_gbr, fraction_ann], 
                                     index = ["Polynomial Regression", "Gradient Boosting", "ANN"],
                                     columns = [operating_points[i]]).transpose()

        fraction = pd.concat([fraction, fraction_vals], axis = 0)

    # Bar charts
    fig, ax = plt.subplots(figsize = (36, 31), tight_layout = True)
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"

    # Data
    x = np.arange(len(operating_points))
    w = 0.25
    y1 = fraction.iloc[:]["Polynomial Regression"]
    y2 = fraction.iloc[:]["Gradient Boosting"]
    y3 = fraction.iloc[:]["ANN"]

    # Plot
    ax.bar(x-w, y1, w, color = "indianred", label = "Polynomial Regression", zorder = 12)
    ax.bar(x, y2, w, color = "darkgreen", label = "Gradient Boosting", zorder = 12)
    ax.bar(x+w, y3, w, color = "royalblue", label = "ANN", zorder = 12)
    ax.set_title("Models' weakness assessment \n Fraction of RMSE over Test data Standard Deviation", 
                 fontsize = 50, fontname = "Times New Roman")
    ax.set_xticks(x, operating_points, fontsize = 42, 
                  fontname = "Times New Roman")
    ax.set_xlabel("Operating points", fontsize = 48, 
                  fontname = "Times New Roman")
    ax.set_ylabel(r'$\frac{RMSE}{\sigma_{test\,data}}$', 
                  fontsize = 48, fontname = "Times New Roman")
    ax.set_ylim(top = 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.2), np.round(np.arange(0, 1.1, 0.2),1), 
                  fontsize = 42, fontname = "Times New Roman")
    ax.axhline(y = 1, color = "r", linestyle = "--")
    ax.legend(fontsize = 40, ncols = 3, bbox_to_anchor = (1.05, -0.2), facecolor = "snow", edgecolor = "k")
    ax.grid(color = "silver", zorder = 0)

    # Add labels on top
    add_labels(operating_points, y1, y2, y3, w)

    plt.show()

except Exception as p:
    raise RuntimeError("Data plotting error:", p)