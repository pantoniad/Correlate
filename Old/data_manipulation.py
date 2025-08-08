import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import correlations_class as corr

class data_manipulation():

    def __init__(self, data_og, clmns, data_range):

        self.data_og = data_og
        self.clmns = clmns
        self.data_range = data_range

    def __repr__(self):
        pass

    def data_extraction(self):
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

        #
        data_og = self.data_og
        clmns = self.clmns
        data_range = self.data_range
        
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

    def data_processing(self, dict, PRoverall):

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
        # Extract conditions from dictionary
        Tbin = dict["Idle"][0]
        Tbout = dict["Idle"][1]
        Pbin = 1.05*dict["Idle"][2]
        Pbout = dict["Idle"][2]
        far = dict["idle"][4]
        m_dot = dict["Idle"][3]
        h = 0
        density = 1.29

        # Generate the class instance
        correlations = corr(Tbin, Tbout, Pbin, Pbout, far, m_dot, h, density)

        # Matrix to save results
        results = np.zeros((len(dict.keys()), 4))
        counter = 0

        # Calculate the Emissions Indices
        for key in dict:
            print(f"Now processing: {key} conditions")
            
            # Novelo
            results[counter, 0] = correlations.novelo()

            # Lewis
            results[counter, 1] = correlations.lewis_nox()

            # Rokke
            results[counter, 2] = correlations.rokke_nox(PRoverall = PRoverall)

            # Kyprianidis
            results[counter, 3] = correlations.kyprianidis()

            counter = counter + 1

        # Round results 
        results = np.round(results, 2)

        return results