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

# Train: Polynomial Regression
parameters = {"Degrees": 2, "Include Bias": True}
polymodel, polyfeatures, train_poly, test_poly = modelsIdle.polReg(
    xtrain = xtrain, ytrain = ytrain, xtest = xdev, ytest = ydev,
    parameters = parameters
)

# Get metrics
metrics = modelsIdle.performance_metrics(train = train_poly, test = test_poly)

print(metrics.head())

# Learning curve
modelsIdle.Learning_curve(model = polymodel, model_features = polyfeatures, operating_point="Idle")

