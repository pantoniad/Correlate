# pip install skorch scikit-learn torch torchvision
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve, LearningCurveDisplay
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP


model = models_per_OP.Model()

df_og = pd.read_csv(r"Databank/ICAO_data.csv", delimiter = ";")

clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
         "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
         "NOx EI App (g/kg)"]

drange = [[61, 169]]

# Clea-up dataframe
df_cleaned = data_process.csv_cleanup(df = df_og, clmns = clmns, drange = drange, reset_index = True, save_to_csv = True, path = "Databank/CFM56data.csv")

## Train Polynomial Regressor per operating point ##

## Idle

# Get data
df_final_idle = data_process.df_former(df_cleaned, clmns = ["Pressure Ratio", "Rated Thrust (kN)"], parameter = "Idle")
df_final_idle["Rated Thrust (kN)"] = df_final_idle["Rated Thrust (kN)"].values.astype(float)*0.07
#print(df_final_idle)

# Features and response
features = df_final_idle.filter(["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)"])
response = df_final_idle["NOx EI Idle (g/kg)"]

# Split the data
X_train, y_train, X_test, y_test = data_process.splitter(
    x = features,
    y = response,
    train_split = 0.5, 
    include_dev = False
)

X_train_t = torch.FloatTensor(X_train.values.astype(float))
X_test_t = torch.FloatTensor(X_test.values.astype(float))

y_train_t = torch.FloatTensor(y_train.values.astype(float))
y_test_t = torch.FloatTensor(y_test.values.astype(float))


# 2) Wrap it with skorch to look like an sklearn estimator
net = NeuralNetClassifier(
    model,
    max_epochs=20,
    lr=1e-3,
    optimizer=torch.optim.Adam,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    iterator_train__shuffle=True,
)

# 3) (Optional) put preprocessing + model in a Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('model', net),
])



# 4) Compute the learning curve
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes = np.linspace(0.1, 1.0, 6)  # 10%..100% of data
scorer = make_scorer(f1_score, average='weighted')  # choose what matters

train_sizes, train_scores, val_scores, fit_times, _ = learning_curve(
    estimator=pipe,
    X=X_train_t, y=y_train_t,                    # <-- your data
    cv=cv,
    scoring=scorer,
    n_jobs=-1,
    train_sizes=train_sizes,
    return_times=True
)

# 5) Plot with sklearn’s helper
LearningCurveDisplay.from_estimator(
    estimator=pipe, X=features, y=response, cv=cv, scoring=scorer, train_sizes=train_sizes
)
plt.tight_layout()
plt.show()
