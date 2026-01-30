# taylor_skillmetrics_example.py
import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm
import pandas as pd

fig = plt.figure(figsize=(7, 7))
# Minimal call; SkillMetrics handles axes, grids and labels.
labels = ["Reference", "Model No.1", "Model No.2", "Model No.3", "Model No.4", "Model No.5"]
sm.taylor_diagram(
    np.array([1.2, 0.7, 0.9, 1.1, 1, 1.4]), # STD   
    np.array([0, 0.5, 0.8, 0.95, 0.6, 0.99]), # CRMSD
    np.array([1, 1.044, 0.722, 0.377, 1, 0.271]),   # R2
    markerLabel=["REF"] + labels,    # labels for legend (first entry is REF)
    markerLegend='on',                 # show legend with labels
    markerlabel = labels,
    markerSize = 15, 
    titleOBS = "Validation Data", markerOBS = "o", colOBS = "r", styleOBS = "-",
    tickRMS = np.round(np.arange(0.2, 2, 0.2),3), colRMS = "b", styleRMS = ":", labelrms = "CRMSD",
    colCOR = "k", styleCOR = "--", widthCOR = 1,
    styleSTD = "-", widthSTD = 1 #, titleRMS = "off", tickRMS = [0.0], showlabelsRMS = "off"
)

plt.title("Taylor Diagram — reference vs. three models", pad=30)
plt.tight_layout()
plt.show()