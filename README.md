# Setup 
create a conda environment 
````
conda create -n fem
conda activate fem
````
Install some packages 
```
conda install numpy 
conda install pandas
conda install -c conda-forge tqdm
conda update scipy
pip install gpyopt
```

Install gurobipy (https://www.gurobi.com/gurobi-and-anaconda-for-windows/)
````
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
````

# Execution
You can run FEM with the following command:
```
cd /home/FEM
python fem.py <dataset> <workload> <marginal> <epsilon_split> <noise_multiple> <samples> <epsilon> 
```
For example
````
python fem.py adult 24 3 0.002 0.05 100 0.1
````

Optimize FEM
````
python hyperparameter_search/fem_bo_search.py adult 32 3 0.1
````

# Results
The following plots show the performace of FEM on dataset 
ADULT and LOANS for 3-way and 5-way marginal queries with a fix workload size of 64.
The solid line represents the average of 5 runs. 
of FEM. 

The first step to reproduce these results is to find the optimal hyperparameters 
using the run_fem_grid_search.sh scrip.
```
sh run_fem_grid_search.sh
```

The results of the grid search were then used to define the parameters in run_fem.sh.
The next step is to run
```angular2
sh run_fem.sh
```


![alt text](images/ADULT_64_3.png)
![alt text](images/ADULT_64_5.png)
![alt text](images/LOANS_64_3.png)
![alt text](images/LOANS_64_5.png)
