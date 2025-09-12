import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings

class FuelFlowMethods:

    """
    FuelFlowMethods: a series of functions containing the Fuel Flow based emissions
                    estimation methods. These methods are provided by DLR, aka DLR
                    Fuel Flow method, and Boeing, aka Boeing Fuel Flow Method 2 or 
                    BFFM2. 
                    The class requires the usage of a pandas dataframe that contains 
                    the data (ICAO fuel flows and EIs for NOx), altitude (in meters)
                    and flight speed (in Mach number) for the four LTO cycle operating
                    points, as defined by ICAO. 
    
    """

    def __init__(self, datapoints: pd.DataFrame, fitting: str, check_fit: str = False):
        """
        self:

        Inputs:
        - datapoints: a dataframe containing the data required for the function. 
                        On the x axis, the operating points are used as keys. 
                        On the y axis, the ICAO retrieved Fuel Flow (kg/s) and 
                        NOx Emissions Indices (gNOx/kgFuel) are given followed by
                        the operating altitude (m) and the flight speed (Mach number)
        - fitting: the type of fitting used on the second step of the DLR or the Boeing
                    fuel flow methods
        """

        self.datapoints = datapoints
        self.fitting = fitting
        self.check_fit = check_fit

    def __repr__(self):
        pass

    def ISA_conditions(alt):
        """
        ISA_conditions:

        Inputs:
        -

        Outputs:
        -

        """
        
        # Ambient conditions calculations = ISA
        Tamb = 288.15 - 6.5*alt/10**3
        Pamb = 101325*(1-0.0065*alt/Tamb)**(5.2561)

        return Tamb, Pamb
        
        
    def dlrFF(self):
        """
        dlrFF:

        Inputs:
        -

        Outputs:
        -
        """

        # Get parameters from self
        data = self.datapoints
        eisICAO = data.iloc[0][:].values.astype(float)
        ffsICAO = data.iloc[1][:].values.astype(float)
        alt = data.iloc[2][:].values.astype(float)
        speed = data.iloc[3][:].values.astype(float)

        # Ambient conditions calculations = ISA
        Tamb = 288.15 - 6.5*alt/10**3
        Pamb = 101325*(1-0.0065*alt/Tamb)**(5.2561)
        
        # Total conditions
        Ttotal = Tamb*(1+0.2*speed)
        Ptotal = Pamb*(1+0.2*speed**2)**3.5

        thetaTotal = Ttotal/288.15
        deltaTotal = Ptotal/101325

        # Step 1 - Reference fuel
        ffref = ffsICAO/(deltaTotal*np.sqrt(thetaTotal))

        # Step 2.1 - Fit the data
        if self.fitting == "Parabolic":

            # Fit the data using a parabolic function
            x = np.sort(ffsICAO)
            y = np.sort(eisICAO)
            z = np.polyfit(x, y, 2)
            pol = np.poly1d(z)

            if self.check_fit == True:
                plt.plot(x, y, "--*",label = "Initial")
                plt.plot(x, pol(x), "-8",label = "Fitted")
                plt.xlabel("Fuel flow (kg/s)")
                plt.ylabel("Emissions index - NOx (gNOx/kgFuel)")
                plt.legend()
                plt.grid(color = "silver", linestyle = ":")
                plt.show()
        
        elif self.fitting == "Third degree":
            
            # Fit the data using a parabolic function
            x = np.sort(ffsICAO)
            y = np.sort(eisICAO)
            z = np.polyfit(x, y, 3)
            pol = np.poly1d(z)

            if self.check_fit == True:
                plt.plot(x, y, "--*",label = "Initial")
                plt.plot(x, pol(x), "-8",label = "Fitted")
                plt.xlabel("Fuel flow (kg/s)")
                plt.ylabel("Emissions index - NOx (gNOx/kgFuel)")
                plt.legend()
                plt.grid(color = "silver", linestyle = ":")
                plt.show()
    
        else:
            warnings.warn("No fitting method selected. Choosing default: Parabolic")

            # Fit the data using a parabolic function
            x = np.sort(ffsICAO)
            y = np.sort(eisICAO)
            z = np.polyfit(x, y, 2)
            pol = np.poly1d(z)

        # Step 2.2 - Get reference value for EI using the reference Fuel Flows
        eiref = pol(ffref)
        #print(eisICAO)

        # Step 3 - EINOx at operating conditions
        altf = 3.2808399*alt # altitude conversion to feet
        omega = 10**(-3)*np.exp(-0.0001426*(altf - 12900)) # omega
        H = -19*(omega-0.00634) # humidity factor

        einox = eiref*deltaTotal**0.4*thetaTotal**3*np.exp(H)

        # Convert to dataframe
        d = {
            "Idle": einox[0],
            "Take-off": einox[1],
            "Climb-out": einox[2],
            "Approach": einox[3]
        }

        einox = pd.DataFrame(
            data = d, 
            index = ["NOx Emissions Index (gNOx/kgFuel)"]
        )

        return np.round(einox,3)

"""
# Usage example

# Data and clmns to keep
data_og = pd.read_csv(r"E:/Correlate/Databank/ICAO_data.csv", delimiter=";")
clmns = ["NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)", "NOx EI App (g/kg)", 
         "Fuel Flow Idle (kg/sec)", "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)", "Fuel Flow App (kg/sec)"]

# EIs and Fuel flow data from ICAO for all engines
eisff = data_og[clmns]
cfm56_range = [[135, 136]]

engineData = eisff.iloc[range(cfm56_range[0][0], cfm56_range[0][1])]

# Operating conditions 
speed = [0, 0.4, 0.4, 0.3]  # Mach number
alt = [0, 11, 304, 905]     # meters

d = {
    "EINOx": engineData.iloc[0][0:4].values.astype(float),
    "Fuel Flows": engineData.iloc[0][4:8].values.astype(float),
    "Flight altitude": alt,
    "Flight Speed": speed
}

datapoints = pd.DataFrame(
    data = d,
    index = ["Idle", "Take-off", "Climb-out", "Approach"]
)
datapoints = datapoints.T
print(datapoints)

ff = FuelFlowMethods(datapoints = datapoints, fitting = "Parabolic", check_fit = False)
einox = ff.dlrFF()
print(einox)
"""