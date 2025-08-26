import matplotlib.pyplot as plt
import numpy as np

# Fake data
x = np.arange(1, 6)
y1 = np.array([2, 4, 1, 5, 3])
y2 = np.array([3, 5, 2, 6, 4])

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, y1, 'o-', label="y1")
ax.plot(x, y2, 's--', label="y2")

ax.set_title("Experiment with Multi-Column Attached Table")
ax.set_xlabel("Trial")
ax.set_ylabel("Value")
ax.legend()

# === Attached table (multi-column) ===
cell_text = [[f"{a}", f"{b}"] for a, b in zip(y1, y2)]
row_labels = [f"T{i}" for i in x]
col_labels = ["y1", "y2"]

table = ax.table(cellText=cell_text,
                 rowLabels=row_labels,
                 colLabels=col_labels,
                 loc="bottom",
                 cellLoc="center")

# Make space for the table
plt.subplots_adjust(bottom=0.25)

plt.show()
