# LPM
This is my code from my Bachelorthesis about groundwater dating using a data set from the Netherlands. 

The code is divided into the shapefree and the analytic Bayesian models. The corresponding Python files are named 'shapefree.py' and 'analytic.py'. Both models use Input functions, which are created in 'input_functions.py' based on the used Tracers. The concentration calculation uses 'models.py' with for PyMC specified models. The outputs can include relativ deviations, relative errors, convergence assesment, and TTDs. The functions are written in 'opti_functions.py' and in 'models.py'. To analyze the chi-squared values of the results the 'chi.py' and 'chi_analytic.py' Files exist. 

The 'DTTDM.py' is the Python implementation of the Discret Transit Time Distribution Model (DTTDM) specified in "Paleoclimate Signals and Groundwater Age Distributions From 39 Public Water Works in the Netherlands; Insights From Noble Gases and Carbon, Hydrogen and Oxygen Isotope Tracers" (Broers et al. 2021). 

The 'optimization.py' File is an old version of the Bayesian models, which includes both the shapefree and the analytic model. 
