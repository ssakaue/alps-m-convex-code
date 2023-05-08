# Faster M-Convex Function Minimization with Predictions: The Laminar-Convex Case

This repository provides the official implementation of [Faster M-Convex Function Minimization with Predictions: The Laminar-Convex Case]().

## Requirements
- [Python](https://www.python.org) (version 3.9.12 or later)
- [NumPy](https://numpy.org/) (version 1.23.2 or later)
- [NetworkX](https://networkx.org/) (version 2.8.6 or later)
- [Gurobi](https://www.gurobi.com/) (version 10.0.1 or later)
- [Gurobipy](https://pypi.org/project/gurobipy/) (version 10.0.1 or later)
- [Pandas](https://pandas.pydata.org/) (version 1.4.4 or later)
- [Matplotlib](https://matplotlib.org/) (version 3.5.3 or later)
- [Seaborn](https://seaborn.pydata.org/) (version 0.12.0 or later)

## Make Instance
Run the codes in `make-instance.ipynb` to generate instance data, which is stored in `data/`.

## Run Experiment
Run the codes in `run-experiment.ipynb`. Results are saved as a single pickle file and stored in `result/`.

This experiment compares the three methods: Learn, Relax, Cold. 
All the methods use `greedy` imported from `utils.py` to get an optimal integer solution. 
Initial feasible solutions $x^\circ$ of `greedy` differ among the three methods as follows. 

- Learn gets $x^\circ$ via the online subgradient descent method over past instances. Projection onto the simplex rescaled by $R$ is computed with `simplex_projection` imported from `utils.py`. 
- Relax gets $x^\circ$ by solving approximate continuous relaxation of a newly arrived instance with `solve_continuous_quadratic` imported from `continuous.py` (implemented with Gurobi).   
- Cold always sets $x^\circ = (R/n)*\mathbf{1}$. 

## Plot Result
Run the codes in `plot-result.ipynb`. 

## License
MIT
