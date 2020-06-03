# Setup 
create a conda environment 
````
conda create -n fem
conda activate fem
````
git clone private-pgm projet
```
cd /home/
git clone https://github.com/ryan112358/private-pgm.git
cd /home/private-pgm
pip install -r requirements.txt
```

Install gurobipy (https://www.gurobi.com/gurobi-and-anaconda-for-windows/)
````
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
````

````
conda install -c conda-forge tqdm
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
