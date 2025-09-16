import numpy as np

# Values from the image (Linear model prediction, p1 column)
linear_model_p1 = np.array([
    296.68165, 28.58521, 58.97671, 263.03392, 109.99101, 191.36681,
    310.79199, 226.12995, 142.55333, 171.85942
])

# Standard deviation
std_dev = np.std(linear_model_p1, ddof=1)  # sample standard deviation

# CRMSD (Centered Root Mean Square Difference)
# CRMSD requires comparison with a reference (observed values). Since not given,
# we calculate CRMSD relative to the mean of the series itself (common practice).
mean_val = np.mean(linear_model_p1)
crmsd = np.sqrt(np.mean((linear_model_p1 - mean_val) ** 2))
pearson_r = np.corrcoef(linear_model_p1, linear_model_p1)[0, 1]

print(f"Linear model: Standard deviatio: {std_dev}, CRMSD: {crmsd}")
print(f"Linear model: pearson coeff: {pearson_r}")
print()

# Predicted values
predicted = np.array([
    311, 55, 60, 302, 87, 152, 297, 235, 165, 136
])

# Standard deviation
std_dev = np.std(predicted, ddof=1)  # sample standard deviation

# CRMSD (Centered Root Mean Square Difference)
# CRMSD requires comparison with a reference (observed values). Since not given,
# we calculate CRMSD relative to the mean of the series itself (common practice).
mean_val = np.mean(predicted)
crmsd = np.sqrt(np.mean((predicted - mean_val) ** 2))
pearson_r = np.corrcoef(predicted, linear_model_p1)[0, 1]

print(f"Predicted values: standard deviatio: {std_dev}, crmsd: {crmsd}")
print(f"Predicted values: pearson coeff: {pearson_r}")
print()

# Values from the image (Real value column)
real_values = np.array([
    287, 40, 68, 256, 115, 190, 
    300, 222, 145, 172
])

# Pearson correlation coefficient between real values and Linear model prediction (p1)
pearson_r = np.corrcoef(real_values, linear_model_p1)[0, 1]
std_dev = np.std(real_values, ddof=1)  # sample standard deviation
mean_val = np.mean(real_values)
crmsd = np.sqrt(np.mean((real_values - mean_val) ** 2))

print(f"Real values: standard deviatio: {std_dev}, crmsd: {crmsd}")
print(f"Real values: Pearson coeff: {pearson_r}")

