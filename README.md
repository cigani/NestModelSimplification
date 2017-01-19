# NestModelSimplification

Step 1: Download required neuron models Example ((https://bbp.epfl.ch/nmc-portal/microcircuit#/metype/L5_TTPC1_cADpyr/details))

Step 2: Place scripts from here into the working NEURON model directory

Step 3: `Run Simulator.py` -- Ensuring Optimize Flag is set to "True" %% This will be very time consuming. After optimization you should be given the optimal sigma and base current (`i_e0`) values required. Edit the following lines (line numbering kept to make it easier to find) in place 
```Python
 97         self.sigmamax = ### Your SigmaMax ###                                                   
 98         self.sigmamin = ### Your SigmaMin ###                                                  
 99         self.i_e0 =  ### Your Optimal Current ###           
```
The sigmavalues are also saved to a pickled dictionary which will be saved to a directory of your choice: 
```Python
313         pickle.dump(sigmas, open(                                               
314             ### YOUR DIRECTOR/PATH HERE ###, "wb"))
```
Now the real data can be collected swap the Optimize flag to "False" (line 358)`Simulator().main(optimize=False)`. And run Simulator again. If you wish to increase the number of runs change (`n`) in the following code (line 335): 
```Python
    self.run_step(130000)
      n = 0
      while n < ### NUMBER OF RUNS YOU WANT ###:
        self.run_step(21000)
         n += 1
 ```
I do not recommend changing the run time. GIFFitting model has well tested data-set minimum sizes, more data will not resut in a significantly better fit.

Step 4: `Run modelfit.py` -- This will fit the data using GIFFittingToolBox 

Step 5: `Run NestGIFModel.py` -- This will take the dictionary of parameters generated by Step 4 and create a basic NEST network


Note: You will need to modify the source somewhat to set paths for data. I have made those locations as obvious as possible ex. `'### FILL IN ###'` should be changed to your path. 

Dataflow: 
-> Simulator will inject a current vector into the NEURON model and save the required current/voltage response to an HDF5 formated file.
--> Modelfit will load that dataset, fit it, and save a pickled dictionary with parameters locally
---> NestGIFModel will use that pickled dictionary to generate the NEST GIF model of choice

Adendum: The majority of the code base is the `CurrentGenerator.py`. It should function stand-alone without modification. The Simulator will require several small edits to ensure paths && Saved data directories are correct. Please be careful that your saved directories are known to the different scripts. 
