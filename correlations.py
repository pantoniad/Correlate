import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

class Correlations:
    """
    Correlations:
    
    """

    
    def __init__(self, Tbin, Tbout, Pbin, Pbout, far, m_dot, h):
        """
        self

        Inputs:
        -

        Outputs:
        -
        """
        self.Tbin = Tbin
        self.Tbout = Tbout
        self.Pbin = Pbin 
        self.Pbout = Pbout
        self.far = far
        self.m_dot = m_dot
        self.h = h

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
        deltaT = self.Tbout - self.Tbin

        # Correlation formulation
        eiTflame = (a+b*np.exp(c*self.Tbin))*np.exp(f*(hsl-h))*(deltaT/deltaTref)**TF
        ei_nox = eiTflame*(self.Pbin/Pcref)**d

        return ei_nox
    
    def deidewig(self, m_dotSL, einoxSL, PbinSL, TbinSL, Ttotin, Ptotin, wfuel, alt = 10000, method = "Index"):
        """
        deidewig: Deidewig proposed a correlation equation based on fuel flow measurements, mainly 
        to enable real-time emissions estimation during flight. For this, a default value was required 
        (Sea Level values of EIs and other parameters). The in-flight parameters are then calculated 
        by two correlation equations, one based on the Emissions Index approach, that uses thermodynamic
        and cycle parameters for the estimation, and one based on the Fuel Flow approach, that uses two
        correction parameters and a corrected fuel flow value.  

        Inputs (self):
        - Pbin: Burner inlet pressure [bar],
        - Tbin: Burner inlet temperature [K],
        - wair: mass flow through the engine core [kg/s]

        Inputs (user):
        - PbinSL: Burner inlet pressure [bar]
        - TbinSL: Burner inlet temperature at sea level [K], 
        - wairSL: Mass flow rate through the engine core at sea level [kg/s]
        - alt: flight altitude [ft]. Default value: 10000 ft/3000 m
        - method: a way to select between the two methods provided by Deidewig. Either "Index" or "FuelFlow"
        - Ttotin: total temperature at the inlet of the engine [K],
        - Ptotin: total pressure at the inlet of the engine [Pa],
        - wfuel: fuel mass flow rate [kg/s]

        Outputs:
        - einox: The emissions index of NOx [g/kgFuel] 

        Source: Deidewig et. al, Methods to assess aircraft engine emissions in flight, 1996, 
        doi: https://elib.dlr.de/38317/

        Additional info:
        """

        # Humidity factor
        Fh = math.exp((6.29 - math.exp(-0.0001443*(alt-12900)))/53.2)
       
        if method == "Index":

            # Stoichiometric flame temperature
            Tstoic = 2281*(np.power(self.Pbin, 0.009375)+0.000178*np.power(self.Pbin, 0.055)*(self.Tbin-298))

            # Emissions index
            einox = einoxSL * (math.exp(65000/Tstoic))/(math.exp(6500/Tstoic))*((self.Pbin)/PbinSL)*(m_dotSL/self.m_dot)*(TbinSL/self.Tbin)*Fh

        elif method == "FuelFlow":
        
            # Pressure correction factor: f 
            delta = Ptotin/101325

            # Temperature correction factor: theta
            theta = Ttotin/288.15

            # Corrected fuel flow
            wfuelCorr = wfuel/(delta*np.sqrt(theta))
            
            # Corrected NOx Emissions Index
            einoxCorr = 45.83*wfuelCorr - 4.12

            # Emissions index
            einox = einoxCorr*theta**3*delta**0.4*Fh 

        return einox

    def touchton(self, t, Tstm, farStoichiometric, w2f):
        """
        touchton: Touchton proposed the use of a more complex correlation equation. It contained
        many parameters (such as combustor inlet conditions, humidity, equivalence ratio and
        constant parameters). Here all temperature expressions are given in [K], all pressure 
        expressions in [Pa] and specific humidity levels in kg H20/kg dry air. More details below 

        Inputs:
        - t: residence time [s],
        - Tstm: Inlet steam temperature [k],
        - farStoichiometric: Stoichiometric fuel-to-air ratio (based on the fuel)  
        - w2f: water-to-fuel ratio

        Outputs:
        - einox: the emissions index of NOx in kgNOx/kgFuel

        Source: 

        Additional info: G.L.Touchton, 1984, An experimentally verified NOx prediction Algorithm
        Incorporating the effects of Steam injection, doi: https://doi.org/10.1115/1.3239647 
        
        """
        # Constants
        a2 = 19020              # [a2] = 1/MPa*s
        b = 0
        c = 0.00381             
        cstm = -1.44*10**(-4)   # [cstm] = K
        ca = 0.489*10**(-4)     # [ca] = 1/K
        d = 6.407*10**5         # [d] = K
        kstm = 1.121            
        eta = 0.0544            
        
        # Reference values
        phi = self.far/farStoichiometric
        Tstm0 = 288.15
        Tbin0 = 700
        h0 = 0
        phi0 = 1 

        # Deltas 
        deltaTstm = Tstm - Tstm0
        deltaTb = self.Tbin - Tbin0
        deltaH = self.h - h0
       
        deltaPhi = phi - phi0

        # Humidity factor
        rhum = np.exp(-19*deltaH)

        # Steam injection factor
        fraction1 = (1+cstm*deltaTstm +ca*deltaTb)/(1+eta*(w2f))*(w2f)
        rstm = np.exp(-kstm*fraction1)

        # Effective residence time
        teff = (t)/(1+b*t)

        # EINOx expression
        fraction2 = np.exp(-c*deltaTb - (d*deltaPhi**2)/self.Tbin) 
        einox = a2*teff*self.Pbin**(0.5)*fraction2*rhum*rstm

        return einox

emissions = Correlations(500, 1490, 20, 19.5, 0.001, 600)

print("Lewis expression: {}".format(emissions.lewis_nox()))
print('Kyprianidis expression: {}'.format(emissions.kyprianidis()))