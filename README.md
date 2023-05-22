# Faster Discrete Convex Function Minimization with Predictions: The M-Convex Case

This repository provides the official implementation of [Faster Discrete Convex Function Minimization with Predictions: The M-Convex Case]().

## Requirements
- [Python](https://www.python.org) (version 3.9.12 or later)
- [NumPy](https://numpy.org/) (version 1.23.2 or later)
- [NetworkX](https://networkx.org/) (version 2.8.6 or later)
- [Pandas](https://pandas.pydata.org/) (version 1.4.4 or later)
- [Matplotlib](https://matplotlib.org/) (version 3.5.3 or later)
- [Seaborn](https://seaborn.pydata.org/) (version 0.12.0 or later)
- [Gurobi](https://www.gurobi.com/) (version 10.0.1 or later)
- [Gurobipy](https://pypi.org/project/gurobipy/) (version 10.0.1 or later)

## Make Instances
Run the codes in `make-instance.ipynb` to make instances. Datasets are stored in `data/`.

## Run Experiments
Run the codes in `run-experiment.ipynb`. Results are saved as a single pickle file and stored in `result/`.

We here compare three methods: Learn, Relax, Cold. 
All the methods use `greedy` imported from `utils.py` to get optimal integer solutions. 
Initial feasible solutions $x^\circ$ of `greedy` differ among the three methods as follows. 

- Learn gets $x^\circ$ via the online subgradient descent method over past instances. Projection onto the simplex rescaled by $R$ is computed with `simplex_projection` imported from `utils.py`. 
- Relax gets $x^\circ$ by solving approximate continuous relaxation of newly arrived instances with `solve_continuous_quadratic` imported from `continuous.py` (implemented with Gurobi).   
- Cold always sets $x^\circ = \frac{R}{n}\cdot\mathbf{1}$. 

## Plot Results
Run the codes in `plot-result.ipynb`. 

## License
MIT
