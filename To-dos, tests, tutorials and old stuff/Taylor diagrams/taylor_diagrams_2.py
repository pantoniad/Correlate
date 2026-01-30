import pandas as pd
import numpy as np
import os
import warnings
import skill_metrics as sm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch as fancy_arrow

std_working = np.array([1.2])
crmsd_working = np.array([1])

fig, ax = plt.subplots(figsize=(12, 11))
fig.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.85)

plt.rcParams.update({
    "font.size": 18,          # default text size
    "axes.titlesize": 30,     # axes titles
    "axes.labelsize": 28,     # x/y labels
    "xtick.labelsize": 24,    # tick labels
    "ytick.labelsize": 24,
    "legend.fontsize": 18,
})
        
#  Working models
sm.taylor_diagram(
    np.array([1.2, 0.7, 0.9, 1.1, 1, 1.4]), # STD   
    np.array([0, 1.044, 0.722, 0.377, 1, 0.271]), # CRMSD
    np.array([1, 0.5, 0.8, 0.95, 0.6, 0.99]),   # R2
    titleOBS = "Validation data", markerOBS = "o", colOBS = "purple", styleOBS = "-",
    labelRMS = "CRMSD", colRMS = 'g',  
    markerLabel = ["Reference", "Model No.1", "Model No.2", "Model No.3", "Model No.4", "Model No.5"], 
    markersize = 15, markerLegend = "on",
    colsCOR = {"grid": "blue", "title": "b", "tick_labels": "b"}, 
)

# STD arrow
plt.arrow(0, -0.08, 0.25, 0, length_includes_head=True, head_width=0.02, head_length=0.02, clip_on = False, color = "k", width = 0.003)
plt.arrow(1.5, -0.08, - 1.05 * max(std_working) + 0.95*1.05 * max(std_working) , 0, length_includes_head=True, 
            head_width=0.02, head_length=0.02, clip_on = False, color = "k", width = 0.003)
ax.text(0.13, -0.13, "Improving along arrow", ha="center", color="k", fontsize = 20)

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
    color="blue"
)
ax.text(1.28, 1.08, "Improving along arrow", ha="center", color="blue", rotation = -45, fontsize = 18)
ax.add_patch(arrow)

plt.title(f"Taylor diagram - Reference vs five random models", pad=60)
plt.tight_layout()
plt.show()