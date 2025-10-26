from GradientBoosting_main import gbr_main

## Secondary inputs
include_learning_curve = False
include_complexity_plot = True
save_results = True

## Primary inputs
primary_inputs = {
    "Idle": {
        "Train split": 0.5,
        "Include development split": False,
        "Number of estimators": 3500,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 1,
        "Subsample size": 0.4
    },
    "T/O":{
        "Train split": 0.5,
        "Include development split": False,
        "Number of estimators": 3500,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 1,
        "Subsample size": 0.4
    },
    "C/O": {
        "Train split": 0.5,
        "Include development split": False,
        "Number of estimators": 1250,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 150,
        "Subsample size": 0.4
    },
    "App": {
        "Train split": 0.5,
        "Include development split": False,
        "Number of estimators": 4000,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 1,
        "Subsample size": 0.4
    }
}

engine_specs = {
    "Pressure Ratio": 27.6,
    "Rated Thrust (kN)": 117,
    "Fuel flow Idle (kg/s)": 0.144,
    "Fuel flow Take-off (kg/s)": 1.25,
    "Fuel flow Climb-out (kg/s)": 1,
    "Fuel flow Approach (kg/s)": 0.348
}

gbr_main(model_structure = primary_inputs, engine_specs = engine_specs, include_learning_curve = include_learning_curve,
         include_complexity_plot = include_complexity_plot, save_results = save_results)