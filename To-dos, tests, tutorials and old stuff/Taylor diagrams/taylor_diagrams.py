# taylor_skillmetrics_example.py
import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm
import pandas as pd

fig = plt.figure(figsize=(7, 7))
# Minimal call; SkillMetrics handles axes, grids and labels.
labels = ["Reference", "Polynomial (2) Regression", "Gradient Boosting", "Artificial Neural Network"]
sm.taylor_diagram(
    np.array([1.2, 0.206, 0.166, 0.207]), # STD   
    np.array([0, 0.24, 0.235, 0.083]), # CRMSD
    np.array([1, 0.47, 0.46, 0.379]),   # R2
    markerLabel=["REF"] + labels,    # labels for legend (first entry is REF)
    markerLegend='on',                 # show legend with labels
    markerlabel = labels,
    titleOBS = "Validation Data", markerOBS = "o", colOBS = "r", styleOBS = "-",
    tickRMS = np.round(np.arange(0.2, 2, 0.2),3), colRMS = "b", styleRMS = ":", labelrms = "CRMSD",
    colCOR = "k", styleCOR = "--", widthCOR = 1,
    styleSTD = "-", widthSTD = 1 #, titleRMS = "off", tickRMS = [0.0], showlabelsRMS = "off"
)

plt.title("Taylor Diagram — reference vs. three models", pad=30)
plt.tight_layout()
plt.show()