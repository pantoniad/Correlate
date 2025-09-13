
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP

# Load data
df = pd.read_csv(r"Databank/ICAO_data.csv", delimiter = ";")

clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
         "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
         "NOx EI App (g/kg)"]

drange = [[61, 169]]

dfCleanUp = data_process(df = df, clmns = clmns, drange = drange)
df = dfCleanUp.csv_cleanup(reset_index = True, save_to_csv = True, path = "Databank/CFM56data.csv")

# Operating point splitting
dIdle = df.drop(columns= ["Fuel Flow T/O (kg/sec)", 
                          "Fuel Flow C/O (kg/sec)",
                          "Fuel Flow App (kg/sec)",
                          "NOx EI T/O (g/kg)",
                          "NOx EI C/O (g/kg)",
                          "NOx EI App (g/kg)"])

dTakeoff = df.drop(columns= ["Fuel Flow Idle (kg/sec)", 
                          "Fuel Flow C/O (kg/sec)",
                          "Fuel Flow App (kg/sec)",
                          "NOx EI Idle (g/kg)",
                          "NOx EI C/O (g/kg)",
                          "NOx EI App (g/kg)"])

dClimbout = df.drop(columns= ["Fuel Flow Idle (kg/sec)", 
                          "Fuel Flow T/O (kg/sec)",
                          "Fuel Flow App (kg/sec)",
                          "NOx EI Idle (g/kg)",
                          "NOx EI C/O (g/kg)",
                          "NOx EI App (g/kg)"])

dApp = df.drop(columns= ["Fuel Flow Idle (kg/sec)", 
                          "Fuel Flow T/O (kg/sec)",
                          "Fuel Flow C/O (kg/sec)",
                          "NOx EI Idle (g/kg)",
                          "NOx EI T/O (g/kg)",
                          "NOx EI C/O (g/kg)"])

## Initialize models class for each operating point
# Idle
features = dIdle.drop(columns = "NOx EI Idle (g/kg)")
response = dIdle["NOx EI Idle (g/kg)"]
modelsIdle = models_per_OP(data = dIdle, 
                           features = features, 
                           response = response)

# Split data
xtrain, ytrain, xdev, ydev, xtest, ytest = modelsIdle.splitter(train_split = 0.6,
                                                               dev_split = 0.2,
                                                               test_split = 0.2)


# Define operating point configuration
op_cfg = {
    "Idle": {
        "ff_col": "Fuel Flow Idle (kg/sec)",
        "ei_col": "NOx EI Idle (g/kg)",
    },
    "Approach": {
        "ff_col": "Fuel Flow App (kg/sec)",
        "ei_col": "NOx EI App (g/kg)",
    },
    "Climb-out": {
        "ff_col": "Fuel Flow C/O (kg/sec)",
        "ei_col": "NOx EI C/O (g/kg)",
    },
    "Take-off": {
        "ff_col": "Fuel Flow T/O (kg/sec)",
        "ei_col": "NOx EI T/O (g/kg)",
    },
}

FEATURES_BASE = ["Pressure Ratio", "Rated Thrust (kN)"]
DEGREE = 2
RANDOM_STATE = 42

rows = []
split_details = []

models = {}

for op_name, cfg in op_cfg.items():
    # Build features/target for this OP
    X = df[FEATURES_BASE + [cfg["ff_col"]]].copy()
    y = df[cfg["ei_col"]].copy()

    # Split 60/20/20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=RANDOM_STATE
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    # Train degree-2 polynomial regression
    pipe = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=DEGREE, include_bias=False)),
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression()),
        ]
    )
    pipe.fit(X_train, y_train)

    # Grid over (OPR, Fuel Flow) while holding Rated Thrust at dev-set median
    opr_col = "Pressure Ratio"
    rt_col = "Rated Thrust (kN)"
    ff_col = cfg["ff_col"]

    rt_fixed = float(np.median(X_dev[rt_col].values))
    opr_min, opr_max = float(X[opr_col].min()), float(X[opr_col].max())
    ff_min, ff_max = float(X[ff_col].min()), float(X[ff_col].max())

    # Create grid
    n_grid = 35
    opr_grid = np.linspace(opr_min, opr_max, n_grid)
    ff_grid = np.linspace(ff_min, ff_max, n_grid)
    OPR, FF = np.meshgrid(opr_grid, ff_grid)

    # Build grid feature matrix (OPR, Rated Thrust fixed, Fuel Flow)
    grid_features = np.column_stack([OPR.ravel(), np.full(OPR.size, rt_fixed), FF.ravel()])
    Z = pipe.predict(grid_features).reshape(OPR.shape)

    # Prepare dev scatter coordinates
    x_scatter = X_dev[opr_col].values
    y_scatter = X_dev[ff_col].values
    z_scatter = y_dev.values

    # Plot (single figure per OP)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(OPR, FF, Z, alpha=0.6, linewidth=0, antialiased=True)
    ax.scatter(x_scatter, y_scatter, z_scatter, s=18)

    ax.set_xlabel("Pressure Ratio (OPR)")
    ax.set_ylabel(f"{ff_col}")
    ax.set_zlabel(f"{cfg['ei_col']}")
    ax.set_title(f"{op_name} — dev points and model surface (Rated Thrust fixed at median ≈ {rt_fixed:.1f} kN)")

    plt.show()


#metrics_df = pd.DataFrame(rows).sort_values("Operating Point").reset_index(drop=True)
#splits_df = pd.DataFrame(split_details).sort_values("Operating Point").reset_index(drop=True)
#print(metrics_df)
#print(splits_df)


