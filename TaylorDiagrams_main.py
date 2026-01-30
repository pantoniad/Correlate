import pandas as pd
import numpy as np
import os
import warnings
import skill_metrics as sm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch as fancy_arrow

# Import complexity plot data
operating_points = ["Idle", "Take-off", "Climb-out", "App"]

variable_of_interest = {"Gradient Boosting": "estimator",
                        "ANN": "LAYERS"}

complexity_path = {"Gradient Boosting": r"E:\Correlate\model_outputs\Run_2025-12-31\Gradient Boosting\ExecutionTime_15-17-51\Gradient Boosting results\Complexity plots results",
                    "ANN": r"E:\Correlate\model_outputs\Run_2025-12-31\ANN\ExecutionTime_15-42-10\ANN results\Complexity plots results"}

surrogates_paths = {"Polynomial Regression": r"E:\Correlate\model_outputs\Run_2025-12-31\Polynomial Regression\ExecutionTime_14-58-35\Polynomial Regression results", 
                    "Gradient Boosting": r"E:\Correlate\model_outputs\Run_2025-12-31\Gradient Boosting\ExecutionTime_15-17-51\Gradient Boosting results", 
                    "ANN": r"E:\Correlate\model_outputs\Run_2025-12-31\ANN\ExecutionTime_15-42-10\ANN results"}

# Data cleaning
try:
    complexity_gbr_save = pd.DataFrame()
    complexity_ann_save = pd.DataFrame()
    surrogates_df_save = pd.DataFrame()

    for op in operating_points:

        print("Now executing: ", op)

        for model, path in surrogates_paths.items():
            filename = f"saved_metrics_{op}.csv"
            surrogates_final_path = os.path.join(path, filename)
            
            if os.path.exists(surrogates_final_path):
                surr = pd.read_csv(surrogates_final_path, delimiter = ",")
                print(f"    {model} data extraction successfull.")

                cols = ["Unnamed: 0", "MAPE"]
                surr = surr.drop(columns = cols, index = 0)
                surr.insert(0, "Operating point", op)
                surr.insert(1, "Model", model)

            else: 
                warnings.warn("No path found for surrogate model metrics.")
            
            surrogates_df_save = pd.concat([surrogates_df_save, surr], axis = 0)
        
        print()
        # Extract variables from complexity plot files
        for model, path in complexity_path.items():
            filename = f"complexity_metrics_test_{variable_of_interest[model]}_{op}.csv"
            final_complexity_path = os.path.join(path, filename)
    
            # Get variable value
            if variable_of_interest[model] == "estimator":
                variable1 = "Tree depth"
            elif variable_of_interest[model] == "LAYERS":
                variable1 = "No. Deep Layers"

            if os.path.exists(final_complexity_path):
                data = pd.read_csv(final_complexity_path, delimiter = ",")
                print(f"    {model} complexity data extraction successfull.")

                # Gradient Boosting
                #if model == "Gradient Boosting":
                iter_var = int(data[variable1].max())
                
                if model == "Gradient Boosting":
                    iter_step = int(iter_var/3)
                elif model == "ANN":
                    iter_step = int(iter_var/5) 

                for i in range(0, iter_var + 1, iter_step):
                    
                    if i == 0:
                        df = data[data[variable1] == i + 1].copy()
                    else:
                        df = data[data[variable1] == i].copy()
                    best_df = df.loc[[df["Test RMSE"].idxmin()]]

                    cols = ["Unnamed: 0", "Test MAPE"]
                    best_df = best_df.drop(columns = cols)
                    best_df.insert(0, "Operating point", op)
                    best_df.insert(1, "Model", model)

                    if model == "Gradient Boosting":
                        complexity_gbr_save = pd.concat([complexity_gbr_save, best_df], axis = 0)
                    elif model == "ANN":
                        complexity_ann_save = pd.concat([complexity_ann_save, best_df], axis = 0)


            else:
                warnings.warn("No path found for complexity plot metrics.")
            
        
        complexity_gbr_save = complexity_gbr_save[complexity_gbr_save["Test R2"] >= 0]
        complexity_gbr_save = complexity_gbr_save.drop_duplicates()
        complexity_gbr_save = complexity_gbr_save.reset_index(drop = True)

        complexity_ann_save = complexity_ann_save[complexity_ann_save["Test R2"] >= 0 ]
        complexity_ann_save = complexity_ann_save.drop_duplicates()
        complexity_ann_save = complexity_ann_save.reset_index(drop = True)
 
        print()

# Stop execution if something is wrong
except Exception as e:
    raise RuntimeError("Data cleaning process error:", e)

# Needed R2, CRMSD, STD for: 1. Reference (Test data), 2. Surrogate models that I chose, 3. Complexity results for GBR and ANN
for op in operating_points:

    # Plotting
    try:
        
        data_to_plot = pd.DataFrame(columns = ["Model", "STD", "CRMSD", "R"])
        for model, path in surrogates_paths.items():
            
            # Working line for chosen surrogates
            working_line_surrogates = surrogates_df_save[surrogates_df_save["Operating point"] == op][surrogates_df_save[surrogates_df_save["Operating point"] == op]["Model"] == model]
            
            # Get STD, CRMSD, R2
            ref = ["Reference", working_line_surrogates["Data Standard Deviation"].values[0], 0, 0]
            ref = pd.DataFrame(ref, index = ["Model", "STD", "CRMSD", "R"], columns = ["Value"]).T

            surr_model_chars = ["Working" + " " + working_line_surrogates["Model"].values[0], working_line_surrogates["Predictions Standard Deviation"].values[0].astype(float), 
                           working_line_surrogates["CRMSD"].values[0].astype(float), working_line_surrogates["Pearson Correlation coefficient"].values[0].astype(float)]
            surr_model_chars = pd.DataFrame(surr_model_chars, columns = ["Value"], index = ["Model", "STD", "CRMSD", "R"]).T

            if model == "Gradient Boosting":

                working_line_complexity = complexity_gbr_save[complexity_gbr_save["Operating point"] == op][complexity_gbr_save[complexity_gbr_save["Operating point"] == op]["Model"] == model]
                
                test_std = np.reshape(working_line_complexity["Test data based prediciton"].values, (-1,1))
                test_crmsd = np.reshape(working_line_complexity["Test CRMSD"].values, (-1,1))
                test_r2 = np.reshape(working_line_complexity["Test Pearson Correlation coefficient"].values, (-1,1))
                model_array = np.full((working_line_complexity.shape[0],1), model, dtype=object)

                complexity_model_chars = np.append(np.append(np.append(model_array, test_std, axis = 1), test_crmsd, axis = 1), test_r2, axis = 1)

            elif model == "ANN":

                working_line_complexity = complexity_ann_save[complexity_ann_save["Operating point"] == op][complexity_ann_save[complexity_ann_save["Operating point"] == op]["Model"] == model]
                
                test_std = np.reshape(working_line_complexity["Test data based prediciton Standard Deviation"].values, (-1,1))
                test_crmsd = np.reshape(working_line_complexity["Test CRMSD"].values, (-1,1))
                test_r2 = np.reshape(working_line_complexity["Test Pearson Correlation coefficient"].values, (-1,1))
                model_array = np.full((working_line_complexity.shape[0],1), model, dtype=object)
                
                complexity_model_chars = np.append(np.append(np.append(model_array, test_std, axis = 1), test_crmsd, axis = 1), test_r2, axis = 1)

            else:

                complexity_model_chars = pd.DataFrame()

            try:
                _ = complexity_model_chars
            except NameError:
                exists = False
                data_to_plot = pd.concat([data_to_plot, ref, surr_model_chars], axis = 0)
            else:
                exists = True
                if len(complexity_model_chars) == 0:
                    data_to_plot = pd.concat([data_to_plot, ref, surr_model_chars])
                else:
                    complexity_model_chars = pd.DataFrame(complexity_model_chars, columns = ["Model", "STD", "CRMSD", "R"], index = working_line_complexity.shape[0]*["Value"])
                    data_to_plot = pd.concat([data_to_plot, ref, surr_model_chars, complexity_model_chars])

        data_to_plot = data_to_plot.drop_duplicates()
        data_to_plot = data_to_plot.reset_index()
        data_to_plot = data_to_plot.drop(["index"], axis = 1)
        data_to_plot = data_to_plot.set_index("Model")
        
        print("Now plotting:", op)

        # Generate diagram
        std_working = np.r_[data_to_plot.iloc[0]["STD"],
                            data_to_plot.loc["Working Polynomial Regression"]["STD"], 
                            data_to_plot.loc["Working Gradient Boosting"]["STD"],
                            data_to_plot.loc["Working ANN"]["STD"]]
        r_working = np.r_[data_to_plot.iloc[0]["R"], 
                           data_to_plot.loc["Working Polynomial Regression"]["R"], 
                           data_to_plot.loc["Working Gradient Boosting"]["R"],
                           data_to_plot.loc["Working ANN"]["R"]]
        crmsd_working = np.r_[data_to_plot.iloc[0]["CRMSD"],
                              data_to_plot.loc["Working Polynomial Regression"][:]["CRMSD"], 
                              data_to_plot.loc["Working Gradient Boosting"]["CRMSD"],
                              data_to_plot.loc["Working ANN"]["CRMSD"]]

        std_gbr = np.r_[data_to_plot.iloc[0]["STD"], data_to_plot.loc["Gradient Boosting"]["STD"].values.astype(float)]
        crmsd_gbr = np.r_[data_to_plot.iloc[0]["CRMSD"], data_to_plot.loc["Gradient Boosting"]["CRMSD"].values.astype(float)]
        r_gbr = np.r_[data_to_plot.iloc[0]["R"], data_to_plot.loc["Gradient Boosting"]["R"].values.astype(float)]
        labels_gbr = [f"GBR{i}" for i in range(1, len(std_gbr))]

        std_ann = np.r_[data_to_plot.iloc[0]["STD"], data_to_plot.loc["ANN"]["STD"].values.astype(float)]
        crmsd_ann = np.r_[data_to_plot.iloc[0]["CRMSD"], data_to_plot.loc["ANN"]["CRMSD"].values.astype(float)]
        r_ann = np.r_[data_to_plot.iloc[0]["R"], data_to_plot.loc["ANN"]["R"].values.astype(float)]
        labels_ann = [f"ANN{i}" for i in range(1, len(std_ann))]

        
        fig, ax = plt.subplots(figsize=(17, 16))
        fig.subplots_adjust(bottom=0.2)

        plt.rcParams.update({
            "font.size": 17,          # default text size
            "axes.titlesize": 30,     # axes titles
            "axes.labelsize": 28,     # x/y labels
            "xtick.labelsize": 26,    # tick labels
            "ytick.labelsize": 26,
            "legend.fontsize": 17,
        })
                
        #  Working models
        sm.taylor_diagram(
            std_working[0], # STD   
            crmsd_working[0], # CRMSD
            r_working[0],   # R2
            titleOBS = "Validation data", markerOBS = "o", colOBS = "purple", styleOBS = "-",
            labelRMS = "CRMSD", markerColors = {"face" : "limegreen", "edge": "k"}, colRMS = 'g', 
            tickRMS = np.round(np.linspace(0, np.round(max(crmsd_working) + 0.5*max(crmsd_working), 3), 6), 2).tolist(),
            markerLabel = ["Reference", "Working Pol.Reg.", "Working GBR", "Working ANN"], markersize = 22, markerSymbol = "*",
            colsCOR = {"grid": "blue", "title": "b", "tick_labels": "b"}, 
            tickCOR = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1],
            axismax=1.05 * max(std_working) 
        )

        sm.taylor_diagram(
            std_working[0:2], # STD   
            crmsd_working[0:2], # CRMSD
            r_working[0:2],   # R2
            titleOBS = "Validation data", markerOBS = "o", colOBS = "purple", styleOBS = "-",
            markerColors = {"face" : "limegreen", "edge": "r"},
            #markerLabel = ["Reference", "Working Pol.Reg."], 
            markersize = 22, markerSymbol = "*",
            overlay = "on"
        )
        
        sm.taylor_diagram(
            np.array([std_working[0], std_working[2]]), # STD   
            np.array([crmsd_working[0], crmsd_working[2]]), # CRMSD
            np.array([r_working[0], r_working[2]]),   # R2
            titleOBS = "Validation data", markerOBS = "o", colOBS = "purple", styleOBS = "-",
            markerColors = {"face" : "yellow", "edge": "r"},
            #markerLabel = ["Reference", "Working GBR"], 
            markersize = 22, markerSymbol = "*",
            overlay = "on"
        )
       
        sm.taylor_diagram(
            np.array([std_working[0], std_working[3]]), # STD   
            np.array([crmsd_working[0], crmsd_working[3]]), # CRMSD
            np.array([r_working[0], r_working[3]]) ,   # R2
            titleOBS = "Validation data", markerOBS = "o", colOBS = "purple", styleOBS = "-",
            markerColors = {"face" : "royalblue", "edge": "r"},
            #markerLabel = ["Reference", "Working ANN"], 
            markersize = 22, markerSymbol = "*",
            overlay = "on"
        )
 
        # Complexity models
        sm.taylor_diagram(
            np.concatenate((std_gbr[0:1], std_gbr[2:4]), axis = 0), # STD   
            np.concatenate((crmsd_gbr[0:1], crmsd_gbr[2:4]), axis = 0), # CRMSD
            np.concatenate((r_gbr[0:1], r_gbr[2:4]), axis = 0),   # R2
            markerColors = {"face" : "yellow", "edge": "k"},
            #markerLabel = ["Reference", *labels_gbr], 
            markersize = 20, markerSymbol = "s",
            overlay = "on"
        )
        
        sm.taylor_diagram(
            np.concatenate((std_ann[0:1], std_ann[2:4]), axis = 0), # STD   
            np.concatenate((crmsd_ann[0:1], crmsd_ann[2:4]), axis = 0), # CRMSD
            np.concatenate((r_ann[0:1], r_ann[2:4]), axis = 0),   # R2
            markerColors = {"face" : "royalblue", "edge": "k"},
            #markerLabel = ["Reference", *labels_ann], 
            markersize = 20, markerSymbol = "8",
            overlay = "on"
        )
        
        # STD arrow
        plt.arrow(0, -0.1*max(std_working), 0.35*max(std_working), 0, length_includes_head=True, head_width=0.015*max(std_working), head_length=0.015*max(std_working), clip_on = False, color = "k", width = 0.003*max(std_working))
        plt.arrow(11.05*max(std_working), -0.1*max(std_working), - 1.05 * max(std_working) + 0.95*1.05 * max(std_working) , 0, length_includes_head=True, head_width=0.015*max(std_working), head_length=0.015*max(std_working), clip_on = False, color = "k", width = 0.003*max(std_working))
        ax.text(0.17*max(std_working), -0.08*max(std_working), "Improving along arrow", ha="center", color="k")
        

        # Correlation arrow
        start = (0.6, 0.99)
        end   = (0.99, 0.6)

        arrow = fancy_arrow(
            posA=start,          # tail
            posB=end,            # head
            arrowstyle="-|>",     # style of the arrow head
            connectionstyle="arc3,rad=-0.15",  # curvature of the line
            mutation_scale=15,   # size of arrow head
            linewidth=1.5,
            transform=ax.transAxes,
            color="darkblue"
        )
        ax.text(0.88*max(std_working), 0.78*max(std_working), "Improving along arrow", ha="center", color="darkblue", rotation = -44)
        ax.add_patch(arrow)
        # Add custom legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='purple', label='Validation Data', markersize=15, linestyle='', markeredgecolor='purple'),
            plt.Line2D([0], [0], marker='*', color='limegreen', label='Working Polynomial Regression', markersize=20, linestyle='', markeredgecolor='k'),
            plt.Line2D([0], [0], marker='*', color='yellow', label='Working Gradient Boosting', markersize=20, linestyle='', markeredgecolor='r'),
            plt.Line2D([0], [0], marker='*', color='royalblue', label='Working ANN', markersize=20, linestyle='', markeredgecolor='r'),
            plt.Line2D([0], [0], marker='s', color='yellow', label='GBR Complexity Variants', markersize=15, linestyle='', markeredgecolor='k'),
            plt.Line2D([0], [0], marker='8', color='royalblue', label='ANN Complexity Variants', markersize=12, linestyle='', markeredgecolor='k'),
        ]

        plt.legend(handles=legend_elements, loc = "lower center", fontsize=15, frameon=True, facecolor = "snow", bbox_to_anchor = (0.5, -0.28), ncol = 3)
        plt.title(f"Taylor diagram - Model comparison - Operating point: {op}", pad=60)
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        raise RuntimeError("Taylor diagram generation error:", e)
    

# Generate Taylor Diagrams
#std = complexity_df_save["Test data based on prediction Standard Deviation"]
#r2 = complexity_df_save["Test R2"]
#crmsd = complexity_df_save["Test CRMSD"]

