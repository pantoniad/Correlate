# Correlate

The present repository coresponds to my thesis, executed under the supervision of Dr. Vasilis Gkoutzamanis, Dr. Kostantions Bollas, PhD candidate Konstantinos Papadopoulos and Professor Anestis Kalfas all part of the Laboratory of Fluid Mechanics and Turbomachinery of the Mechanical Engineering Departhment of the Aristotle University of Thessaloniki.

The main scope on this repository, and the thesis in general, is to curate the examine the predictive performance of two methodologies, the use of semi-empirical methods and data-driven surrogate models, when estimating the amount of NOx produced from a working aircraft engine. 

This tool focuses on the below sub-areas:
1. Research on how correlation equations are formulated, used and why they were needed in the first place. This includes a thorough literature review background on how aircraft engine combustors work, what parameters (engine and non-engine related) are most important for the estimation of pollutants, and NOx in particular and a collection of multiple correlation equations, retrieved from open literature, that are used to estimate a series of pollutants, like CO2 and CO and not just NOx,
2. Research on the workings of data-driven surrogate model formulation. For this particular thesis, three models have been examined: Polynomial Regression, Gradient Boosting Regression (GBR) using Decision Trees as a base learner and Artificial Neural Networks (ANNs) with Fully Connected Layers (FCLs). 
3. Efficient and consice plotting structures that allow for comprehensive visualization of the results, i.e. Distribution plots 

For the above, the ICAO Emissions Databank has been used for data extraction while experimental data have been retrieved from literature in the form of Emissions Index values per operating point.

For the thesis the CFM56 engine family was studied, while engine specific predicitions were made for the CFM56-7B26 engine. Data for the particular model have been retrieved both from open literature, the documentation available from the manufacturer and the ICAO Emissions Databank.

The performance tool used for 0D analysis of the engine and, mainly, to extract engine and operating-point specific thermodynamic parameters and values was the AeroEngineS software. This tool has been developed in-house, by fellow LFMT students that conducted their thesis. 

To cite this repository, please include the below credentials:
- Author: Antoniadis Panagiotis,
- e-mail: pantoniad@meng.auth.gr

