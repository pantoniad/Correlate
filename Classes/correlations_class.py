import numpy as np
import math

class Correlations:

    def __init__(self, Tbin, Tbout, Pbin, Pbout, far, m_dot, h, desnity):
        """
        self

        Inputs:
        - Tbin, Tbout: burner inlet and outlet temperatures (K),
        - Pbin, Pbout: burner inlet and outlet pressures (Pa),
        - far: Fuel-to-air ratio,
        - m_dot: air mass flow rate (kg/s),
        - density: density of inlet air (kg/m3)

        Outputs:
        - self

        """
        self.Tbin = Tbin
        self.Tbout = Tbout
        self.Pbin = Pbin 
        self.Pbout = Pbout
        self.far = far
        self.m_dot = m_dot
        self.h = h
        self.density = desnity

    def __repr__(self):
        pass
    

    
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
        - nox: amount of NOx produced in [kgNOx/kgfuel]

        Issues:
        - Converted to EInox values of nox are way smaller than the ones calculated from other correlations
        - No knowledge of the unit of pressure. 

        Source: Gas turbine combustion: Alternative fuels and emissions (IBSN-13: 978-1-4200-8605-8)
        """
    
        # Base expression
        nox = 3.32*10**(-6)*np.exp(0.008*self.Tbout)*(self.Pbout)**0.5
    
        # Convert to Emissions Index (EI - kg of pollutant / kg of fuel)
        ei_nox = 10**3*(nox*46.01*self.far)/(10**6*28.3)

        return ei_nox 
    
    def rokke_nox(self, PRoverall, method = "Index"):
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
        - m_dot: air mass flow rate (kg/s)
        - FAR: Fuel-to-Air ratio

        Outputs:
        - nox: the amount of produced NOx in EI format (kgNOx/kgFuel) or parts per million (ppmv)

        Source: Rokke et. al - Pollutant emissions from gas fired turbine engines in offshore practice 
                Doi: https://doi.org/10.1115/93-GT-170
        Additional info: A PPMV and an EI version are provided in the paper
        """

        # Base expression
        if method == "Index":
            ei_nox = 1.46*np.power(PRoverall, 1.42)*np.power(self.m_dot, 0.3)*np.power(self.far, 0.72)
        elif method == "PPMV":
            nox_ppmv = 18.1*np.power(PRoverall, 1.42)*np.power(self.m_dot, 0.3)*np.power(self.far, 0.72)
            ei_nox = nox_ppmv*46.01*self.far/(10**(6)*28.3) 

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
        - ei_nox: Emissions index of NOx (kgNOx/kgFuel)

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
        ei_nox = eiTflame*(self.Pbin*10**(-3)/Pcref)**d

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
        - Pbin: Burner inlet pressure [Pa] - Conversion to [bar] included,
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
        - einox: The emissions index of NOx [kg/kgFuel] 

        Source: Deidewig et. al, Methods to assess aircraft engine emissions in flight, 1996, 
        doi: https://elib.dlr.de/38317/

        Additional info:
        """

        # Pa to bar conversion
        Pbin = 1*10**(-5)*self.Pbin

        # Humidity factor
        Fh = math.exp((6.29 - math.exp(-0.0001443*(alt-12900)))/53.2)
       
        if method == "Index":

            # Stoichiometric flame temperature
            Tstoic = 2281*(np.power(Pbin, 0.009375)+0.000178*np.power(Pbin, 0.055)*(self.Tbin-298))

            # Emissions index
            einox = 1e3*einoxSL * (math.exp(67500/Tstoic))/(math.exp(6750/Tstoic))*((Pbin)/PbinSL)*(m_dotSL/self.m_dot)*(TbinSL/self.Tbin)*Fh

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
            einox = 1e3*einoxCorr*theta**3*delta**0.4*Fh 

        return einox

    def touchton(self, t, Tstm, farStoichiometric, w2f):
        """
        touchton: Touchton proposed the use of a more complex correlation equation. It contained
        many parameters (such as combustor inlet conditions, humidity, equivalence ratio and
        constant parameters). Here all temperature expressions are given in [K], all pressure 
        expressions in [Pa] and specific humidity levels in kg H20/kg dry air. More details below 

        Inputs:
        - t: residence time [s],
        - Tstm: Inlet steam temperature [K],
        - farStoichiometric: Stoichiometric fuel-to-air ratio (based on the fuel)  
        - w2f: water-to-fuel ratio

        Outputs:
        - einox: the emissions index of NOx in kgNO2/kgFuel

        Source: 

        Additional info: G.L.Touchton, 1984, An experimentally verified NOx prediction Algorithm
        Incorporating the effects of Steam injection, doi: https://doi.org/10.1115/1.3239647 
        
        """
        # Constants
        a2 = 19020              # [a2] = 1/MPa*s
        b = 0
        c = 0.00381             # [c] = 1/K
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
        einox = 1e3*a2*teff*self.Pbin**(0.5)*fraction2*rhum*rstm

        return einox
    
    def becker(self, Tfl, method = "simplified"):
        """
        becker: Becker, in the referenced paper with Perkavec, proposed two correlation
        equations. The first is a simpler (method = "simplified") expression, taking into 
        consideration combustor inlet pressure and the adiabatic flame temperature. The 
        advanced correlation (method = "advanced") makes use of more paremeters

        Inputs:
        - Tfl: flame temperature (Simplified) or adiabatic flame temperature (Advanced) 
        - method: 
            - "simplified": use the simpler version of Becker's proposed correlation equation
            - "advanced": use the complex version, as proposed in the paper 

        Outputs:
        - ei_nox: NOx emissions in kgNOx/kgfuel, dry air conditions for the advanced method

        Source: T. Becker et. al, 1994, The capability of different semianalytical equations for
        estimation of NOx emissions of gas turbines, doi: https://doi.org/10.1115/94-GT-282
 
        """

        if method == "simplified":

            # NOx expression - Parameters can be changed based on tabel 3
            nox_ppmv = 5.73*10**(-6)*np.exp(0.00833*Tfl)*(self.Pbin*10**(-3))**0.5  
            ei_nox = 10**3*nox_ppmv*46.01*self.far/(10**(6)*28.3) 

        elif method == "advanced":
            
            # nox_ppmv assigned to none
            nox_ppmv = None

            # ISO conditions. Taken from Table 1, averaged values
            TrISO = 600 
            m_dotISO = 70 # [kg/s]
            PrISO = 8*101325
            TfirISOfl = 1400
            Tfirpl = 1200

            # Other parameters
            s2f = 0.8
            v_dot = self.density*self.m_dot
            m_dotFuel = self.far*self.m_dot
            
            # Specific NOx emission index
            sNOx = 11000 # average value from table 3, mgNOx/kgfuel
            
            # Corrected specific NOx 
            sNOxCorr = sNOx*(1-0.3571*s2f) 

            # Residence time
            ft = (m_dotISO*self.Pbin*TrISO)/(self.m_dot*PrISO*self.Tbout)

            # Geometrical expression: TfirISOfl: firing temperature at ISO condition, full load, Tfl_pl: firing temperature, part load
            fTL = (Tfl-TfirISOfl)/(Tfl-Tfirpl)

            # Advanced NOx expression
            exponential = np.exp(Tfl*(1+(self.Pbin**0.5-PrISO**0.5)/100-2208)/247.7)
            fraction = m_dotFuel/v_dot
            
            # NOx in mg/Nm3 
            nox = sNOxCorr*exponential*fraction*ft*fTL

            # Nm3 to kg/kg. Assuming normal conditions at 0 degrees celcius or 273.15K
            ei_nox = 7.73*10**(-4)*nox

        else:
            print("No method given")

        return ei_nox
       

    def odgers(self, Tfl, t):
        """
        odgers:

        Inputs:
        - Tfl: flame temperature [K]
        - t: residence time [s]

        Ouputs:
        - einox: emissions index of NOx [kgNOx/kgFuel]

        Source: Becker et. al, 1994, The capability of different semi-analytical equations for estimation of NOx 
        emissions of Gas Turbines, doi: https://doi.org/10.1115/94-GT-282
        
        """
        
        einox = 29*np.exp(-2167/Tfl)*self.Pbin**0.66*(1-np.exp(-250*t))

        return einox 
    
    def perkavec(self):
        """
        perkavec: 

        Inputs:
        - Pbin: combustor inlet pressure (Pa), 
        - Tbin: combustor inlet temperature (K),
        - far: fuel-to-air ratio
        - m_dot: mass flow rate through the engine (kg/s)

        Outputs:
        - nox: NOx emissions in mg/Nm3, dry

        Source: Becker et. al, 1994, The capability of different semi-analytical equations for estimation of NOx 
        emissions of Gas Turbines, doi: https://doi.org/10.1115/94-GT-282

        """

        # Absolute humidity
        hAbs = 0
        hAbsS = 0
        
        # NOx in mg/Nm3
        nox = 8.28*self.Pbin**0.5*self.far**1.4*self.m_dot**(-.22)*np.exp(self.Tbin/260)*np.exp(-58*(hAbs-hAbsS))

        # Assuming normal conditions at 0 degrees Celcius or 273.15K
        einox = 7.73*10**(-4)*nox

        return einox
    
    def lefebvre(self, Vc: float, Tpz: float, Tst: float):
        """
        lefebvre: correlation equation proposed by Lefebvre and reported on by other authors. 
                    It takes into consideration the combustor inlet conditions and characteristics
                    (Vc, Tpz) and also fuel related parameters (Tst). All units in SI, output
                    in gNOx/kgFuel

        Inputs:
        - self,
        - Vc: Combustor volume, float, m3,
        - Tpz: Primary combustion zone temperature, float, K,
        - Tst: stoichiometric combustion temperature, float, K

        Outputs:
        - einox: emissions index of NOx, gNOx/kgFuel

        Source: Sascha Kaiser et. al, 2022, The water enhanced turbofan as enabler for climate-neutral aviation, https://doi.org/10.3390/app122312431 
        """

        # Data retrieval from self
        Pin = self.Pbin
        Tout = self.Tbout
        mdotin = self.m_dot
        
        # Correlation equation parts
        part1 = (Pin*10**(-3)*Vc)/(mdotin*Tout)
        part2 = np.exp(0.01*Tst)
        part3 = (Pin*10**(-3))**0.25
        
        # Correlation
        einox = 9*10**(-8)*part1*part2*part3

        return einox

    def gasturb(self, WAR: float = 0):
        """
        gasturb: Correlation equation proposed and used by GasTurb software. Takes into
                    consideration combustor inlet parameters and the humidity level (WAR).

        Inputs:
        - self:,
        - WAR: Water-to-Air ratio, float, non-dimensional, values from 0-1 (0-100%), default 0

        Outputs:
        - einox: Emissions Index of NOx, gNOx/kgFuel

        Source: Sascha Kaiser et. al, 2022, The water enhanced turbofan as enabler for climate-neutral aviation, https://doi.org/10.3390/app122312431 
        """

        # Data retrieval from self
        Pin = self.Pbin
        Tin = self.Tbin

        # Correlation break-down
        part1 = np.exp((Tin-826)/194)
        part2 = (Pin/(2.965*10**6))**0.4
        part3 = np.exp((6.29-10**3*WAR)/53.2)

        # Correlation equation
        einox = 32*part1*part2*part3

        return einox
    
    def generalElectric(self, WAR: float = 0):
        """
        generalElectric: A correlation equation proposed by General Electric.
                        Takes into consideration the combustor inlet conditions
                        and the humidity level through the Water-to-Air ratio input 

        Inputs:
        - self,
        - WAR: Water-to-Air ratio, float, non-dimensional, values from 0-1 (0-100%), default 0

        Outputs:
        - einox: Emissions Index of NOx, gNOx/kgFuel

        Source: Sascha Kaiser et. al, 2022, The water enhanced turbofan as enabler for climate-neutral aviation, https://doi.org/10.3390/app122312431 
        
        """

        # Data retrieval from self
        Tin = self.Tbin
        Pin = self.Pbin

        # Correlation break-down
        part1 = np.exp(Tin/194.4)
        part2 = (Pin/101325)**0.4
        part3 = np.exp(-10**3*WAR/53.2)

        # Correlation
        einox = 2.2+0.12425*part1*part2*part3

        return einox
    
    def aeronox(self, Vc: float, R: float = 287):
        """
        aeronox: Correlation equation reported on by multiple authors (Tsalavouta, Sascha).
                It features combustor characteristics, such as combustor volume and inlet
                and outlet conditions. Also takes into account the gas used, through the 
                selection of R 

        Inputs:
        - self,
        - Vc: combustor volume, float, m3,
        - R: specific gas constant, float, J/kg/K, default for air: 287

        Outputs:    
        - einox: Emissions Index of NOx, gNOx/kgFuel

        Source: Sascha Kaiser et. al, 2022, The water enhanced turbofan as enabler for climate-neutral aviation, https://doi.org/10.3390/app122312431 
        """

        # Parameter retrieval from self
        Pin = self.Pbin
        Tin = self.Tbin
        mdotin = self.m_dot
        Tout = self.Tbout

        # Correlation break-down
        part1 = ((Vc*Pin*10**3)/(mdotin*R*Tin))**0.7
        part2 = np.exp(-600/Tout)

        # Correlation equation
        einox = 1.5*part1*part2

        return einox 


# Testing
#emissions = Correlations(500, 1490, 20, 19.5, 0.001, 600, 0.5,1.293)

#print(Correlations.__repr__)
#print("Becker expression: {}".format(emissions.becker(1490, method = "advanced")))
#print('Kyprianidis expression: {}'.format(emissions.kyprianidis()))