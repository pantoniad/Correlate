import numpy as np

FAR = 0.0144
Pc = 3156.71
m_dot = 32.32

# Base expression
nox = 1.46*np.power(32.7, 1.42)*np.power(m_dot, 0.3)*np.power(FAR, 0.72)
print(f"NOx (PPMV): {np.round(nox,2)}")

# Convert to Emissions Index (EI - kg of pollutant / kg of fuel)
ei_nox = (nox*46.01*FAR)/(10**6*29)
print(f"EINOx: {ei_nox}")