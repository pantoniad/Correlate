import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Coefficients (from the provided model)
b0 = 16.542
b1, b2, b3 = -3.279, 3.58, 2.176
b4, b5, b6 = 24.011, -2.379, -45.722
b7, b8, b9 = -3.235, 7.541, 20.098

# Updated ranges
x1_min, x1_max = 10, 28.0
x2_min, x2_max = 0.0, 120.0
x3_fixed = 1.0

# Surface grid
x1 = np.linspace(x1_min, x1_max, 120)
x2 = np.linspace(x2_min, x2_max, 160)
X1, X2 = np.meshgrid(x1, x2)
X3 = x3_fixed

# Polynomial surface
Y = (
    b0
    + b1 * X1 + b2 * X2 + b3 * X3
    + b4 * X1 * X2 + b5 * X1 * X3 + b6 * X2 * X3
    + b7 * (X1 ** 2) + b8 * (X2 ** 2) + b9 * (X3 ** 2)
)

# Create figure
fig = plt.figure(figsize=(11, 8.5))
ax = fig.add_subplot(111, projection="3d")

# Surface
ax.plot_surface(X1, X2, Y, linewidth=0, antialiased=True, alpha=0.9)

# Red solid line at x1 = 22.5
x1_line = 27.6
x2_line = np.linspace(x2_min, x2_max, 200)

Y_line = (
    b0
    + b1 * x1_line + b2 * x2_line + b3 * X3
    + b4 * x1_line * x2_line + b5 * x1_line * X3 + b6 * x2_line * X3
    + b7 * (x1_line ** 2) + b8 * (x2_line ** 2) + b9 * (X3 ** 2)
)

ax.plot(
    np.full_like(x2_line, x1_line),
    x2_line,
    Y_line,
    linewidth=10
)

# Labels and title
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title("2nd-degree polynomial surface (x3 = 1.0)\nRed line: x1 = 22.5")

# View angle
ax.view_init(elev=25, azim=-135)

# Save
plt.tight_layout()
plt.show()

