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
```

Install gurobipy (https://www.gurobi.com/gurobi-and-anaconda-for-windows/)
````
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
````

````
cd /home/FEM
````

# Execution
You can run FEM with the following command:
```

python fem.py <dataset> <workload> <marginal> <eps0> <noise> <samples> <epsilon> 
```
For example
````
python fem.py adult 24 3 0.003 2 100 0.1
````
