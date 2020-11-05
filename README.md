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
python fem.py adult 24 3 100 50 100 0.1
````

Optimize FEM
````
python hyperparameter_search/fem_bo_search.py adult 32 3 0.1
````
