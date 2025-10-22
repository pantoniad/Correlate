import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

class Correlations:
    """
    Correlations:
    
    """

    
    def __init__(self, Tbin, Tbout, Pbin, Pbout, far, m_dot):
        
        self.Tbin = Tbin
        self.Tbout = Tbout
        self.Pbin = Pbin 
        self.Pbout = Pbout
        self.far = far
        self.m_dot = m_dot

    def novelo(self):
        """
        novelo function: contains the expression for the calculation of 
        NOx based on the Burner exit temperature.

        Inputs: 
        - Tb: burner inlet temperature [K]

        Outputs:
        - einox: emissions index of NOx (g NOx/kg fuel)

        Source: DOI:10.1016/j.trd.2018.12.021
        """

        einox = 0.000175*self.Tbin**2-0.14809*self.Tbin+35.09184

        return einox
    
    def lewis_nox(self):
        """
        lewis_nox function: contains the correlation equation used to 
        calculate the emissions index of NOx based on the research 
        conducted by Lewis. 

        Inputs: 
        - Tc: combustor exit temperature [K]
        - Pc:  combutor exit pressure [Pa]
        - FAR: Fuel-to-Air ratio

        Outputs:
        - nox: amount of NOx produced in Particles Per Million (PPMV)

        Issues:
        - Converted to EInox values of nox are way smaller than the ones calculated from other correlations
        - No knowledge of the unit of pressure. 

        Source: Gas turbine combustion: Alternative fuels and emissions (IBSN-13: 978-1-4200-8605-8)
        """
    
        # Base expression
        nox = 3.32*10**(-6)*np.exp(0.008*self.Tbout)*(self.Pbout*10**(-2))**0.5
    
        # Convert to Emissions Index (EI - kg of pollutant / kg of fuel)
        ei_nox = (nox*46.01*self.far)/(29)

        return ei_nox 
    
    def rokke_nox(self,PRoverall):
        """
        rokke_nox: contains the correlation equation developed by Rokke et al. and
        provided by Lefebvre on its book "Gas Turbine combustion: Alternative Fuels and 
        emissions". It correlates the amount of NOx produced (ppmv) with the combustion pressure,
        the inlet air mass flow rate and the Fuel-to-Air ratio

        Based on:
        - Gas powered industrial engines,
        - Power outputs from 35-100%. Not suitable for idle conditions

        Inputs:
        - PRoverall: pressure ratio of the compressor of the engine
        - m_dot: air mass flow rate
        - FAR: Fuel-to-Air ratio

        Outputs:
        - nox: the amount of produced NOx in part per million by volume (ppmv)

        Source: Rokke et. al - Pollutant emissions from gas fired turbine engines in offshore practice 
                Doi: https://doi.org/10.1115/93-GT-170
        Additional info: A PPMV and an EI version are provided in the paper
        """

        # Base expression
        ei_nox = 1.46*np.power(PRoverall, 1.42)*np.power(self.m_dot, 0.3)*np.power(self.far, 0.72)

        return ei_nox

def kyprianidis(self, h = 0):
    """
    kyprianidis: Correlation equation based on data derived from a large amount of engine 
    performance data produced using an in-house engine performance tool and measurements 
    taken from the ICAO Emissions Databank. The predictive capability of this correlation 
    has been verified within the project NEWAC.

    Inputs:
    - Tcin: Combustor inlet temperature (K)
    - Tcout: Combustor outlet temperature (K)
    - Pcin: Combustor inlet pressure (Pa)
    - h: Ambient humidity (kg H20/kg dry air). Default: h = 0

    Outputs:
    - ei_nox: Emissions index of NOx (gNOx/kgFuel)

    Source: On the trade-off between aviation NOx and energy efficiency, K.G.Kyprianidis
            DOI: https://doi.org/10.1016/j.apenergy.2015.12.055

    Additional info:
    
    """
    # Parameter definition
    a = 8.4
    b = 0.0209
    c = 0.0082
    d = 0.4
    f = 19
    TF = 0
    Pcref = 3000 #kPa
    deltaTref = 300
    hsl = 0.006344 

    # Additional parameters
    deltaT = self.Tcout - self.Tcin

    # Correlation formulation
    eiTflame = (a+b*np.exp(c*self.Tcin))*np.exp(f*(hsl-h))*(deltaT/deltaTref)**TF
    ei_nox = eiTflame*(self.Pcin/Pcref)**d

    return ei_nox