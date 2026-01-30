# taylor_skillmetrics_example.py
import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm
import pandas as pd
from matplotlib.patches import FancyArrowPatch as fancy_arrow

fig, ax= plt.subplots(figsize=(12,10))
plt.rcParams.update({
            "font.size": 16,          # default text size
            "axes.titlesize": 28,     # axes titles
            "axes.labelsize": 26,     # x/y labels
            "xtick.labelsize": 24,    # tick labels
            "ytick.labelsize": 24,
            "legend.fontsize": 15,
        })
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

# STD arrow
plt.arrow(0, -0.1, 0.5, 0, length_includes_head=True, head_width=0.023, 
          head_length=0.023, clip_on = False, color = "k", width = 0.0025) # forward arrow
plt.arrow(1.5, -0.1, -0.05, 0, length_includes_head=True, head_width=0.023, 
          head_length=0.023, clip_on = False, color = "k", width = 0.0025) # backward arrow
ax.text(0.25, -0.17, "Improving along arrow", ha="center", color="k")

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
ax.text(1.27, 1.08, "Improving along arrow", ha="center", color="blue", rotation = -45)
ax.add_patch(arrow)

plt.title("Taylor Diagram — reference vs. five random models", pad=60)
plt.tight_layout()
plt.savefig(r"E:\Research files\Thesis manuscript\pictures\typical_taylor.png")
plt.show()