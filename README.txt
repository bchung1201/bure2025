All required packages listed in requirements.txt may be installed through pip.

 There is one .py file for each experiment-related figure in the paper. Each file contains the basic simulation code with the necessary overheads to keep track of the relevant data from the simulation. To generate a figure, simply run the corresponding .py file, e.g. "python figfour_a.py" for figure 4a. The simulation code relies on the same basic EXP3 algorithm applied to the network model described in the paper. The only differences in implementation between the different files relate to keeping track of data relevant to the figure, e.g. clearing rates, joint distributions of strategies, or particular values for the number of servers and queues.

 Note: depending on hardware, some of the simulations can take up to a few hours to run.
