import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Functions
"""
def data_extraction(data_og, clmns, data_range):
    """
    data_extraction: the purpose of this function is to extract the data based from
    the complete data set, based on the data ranges provided in the data_range list.

    Inputs:
    - data (dataframe): the data  from which the extraction happens, in a dataframe
            format
    - clmns (list of strings): the collumn names that we want to keep from the complete
            dataframe
    - data_range (list): the range of the rows that we want to keep 

    Outputs:
    - data_avg (numpy array): the averaged, by collumn, data
    - data_d (dataframe): the part of the complete data set that we wanted
    """ 
    
    # Archive the output variable
    partition = np.empty((len(clmns), len(data_range)))
    counter1 = 0

    # Partition the data based on the ranges given
    for i in data_range:

        # Get starting and ending point of the dataframe
        start = i[0]
        end = i[1]

        # Extract data and reset index
        data = data_og.iloc[start:end]
        data = data.reset_index()
        
        # Drop unwanted collumns
        data_d = data[clmns].astype(float)

        # Calcualte the averages for the selected range start-end
        counter2 = 0
        for j in clmns:
            # Get values to average
            item = data_d.get(j)
            values = item.values
            avg = np.array([sum(values)/len(values)])

            # Assign values to partition list 
            partition[counter2][counter1] = avg

            # AUpdate internal counter
            counter2 = counter2 + 1        

        # Update external counter
        counter1 = counter1 + 1

    # Create dictionary for export 
    data_avg = dict(zip(clmns, np.round(partition,2)))

    return data_avg 

def data_processing(dict, PRoverall):

    """
    data_processing:

    Inputs:
    - dict: dictionary that contains the values for the variables for 
            each flight phase

    Outputs:
    - results: a 4x3 array that contains the results of using the correlations equations
                (size is dependent on number of correlation equations and simulation points
                retrieved from the input dictionary)
    """
    # Matrix to save results
    results = np.zeros((len(dict.keys()), 4))
    counter = 0

    # Calculate the Emissions Indices
    for key in dict:
        print(f"Now processing: {key} conditions")
        
        # Novelo
        results[counter, 0] = novelo(dict[key][0])

        # Lewis
        results[counter, 1] = lewis_nox(dict[key][1], dict[key][2], dict[key][4])

        # Rokke
        results[counter, 2] = rokke_nox(PRoverall, dict[key][3], dict[key][4])

        # Kyprianidis
        results[counter, 3] = kyprianidis(dict[key][0], dict[key][1], dict[key][2] + 0.05*dict[key][2])

        counter = counter + 1

    # Round results 
    results = np.round(results, 2)

    return results


def novelo(Tb):
    """
    novelo function: contains the expression for the calculation of 
    NOx based on the Burner exit temperature.

    Inputs: 
    - Tb: burner inlet temperature [K]

    Outputs:
    - einox: emissions index of NOx (g NOx/kg fuel)

    Source: DOI:10.1016/j.trd.2018.12.021
    """

    einox = 0.000175*Tb**2-0.14809*Tb+35.09184

    return einox

def lewis_nox(Tc, Pc, FAR):
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
    nox = 3.32*10**(-6)*np.exp(0.008*Tc)*(Pc*10**(-2))**0.5
    
    # Convert to Emissions Index (EI - kg of pollutant / kg of fuel)
    ei_nox = (nox*46.01*FAR)/(29)

    return ei_nox

def rokke_nox(PRoverall, m_dot, FAR):
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
    ei_nox = 1.46*np.power(PRoverall, 1.42)*np.power(m_dot, 0.3)*np.power(FAR, 0.72)

    return ei_nox

def kyprianidis(Tcin, Tcout, Pcin, h = 0):
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
    deltaT = Tcout - Tcin

    # Correlation formulation
    eiTflame = (a+b*np.exp(c*Tcin))*np.exp(f*(hsl-h))*(deltaT/deltaTref)**TF
    ei_nox = eiTflame*(Pcin/Pcref)**d

    return ei_nox

def lefebvre_co(Tpz, P3, Vc, deltaP, m_dot, Do, rho, lamda_eff):
    """
    lefebvre_co: contains the correlation equation used to calculate the amount of 
    carbon monoxide produced during the operation of an aicraft engine. It correlates 
    various parameters, such as thermodynamic conditions (P, T), mass flow rate and 
    evaporation volume

    Inputs:
    - Tpz:
    - P3:
    - Vc:
    - deltaP:
    - m_dot:
    - Do:
    - rho:
    - lamda_eff:

    Outputs:
    - eico:

    Sources:
    - Gas turbine combustion: Alternative fuels and emissions (IBSN-13: 978-1-4200-8605-8)
    - Lefebvre - Fuel effects on gas turbine combustion-liner temperature, pattern factor and
    pollutant emissions (https://doi.org/10.2514/3.45059)  
    """

    # Evaporation volume
    Ve = 0.55*m_dot*(Do**2)/(rho*lamda_eff)

    # Calculation of CO emissions index
    eico = (86*m_dot*Tpz**(-0.00345*Tpz))/(Vc-Ve*(deltaP/P3)**0.5*P3**15)

    return eico
    
def plotting(plot_style, x_axis, y_axis, fig_size):
    """
    ei_plotting:
    
    """

    fig0, axs0 = plt.subplots( figsize=(8, 6))
    plt.plot(x_axis, x_axis, marker='o')
    plt.grid(True)
    fig0.suptitle("Emissions Index - CFM56 - ICAO emissions databank", fontsize=14)
    plt.savefig(r"E:/Computational_results/Plots/EIdatabank.png")
