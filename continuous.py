import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import itertools

def solve_continuous_quadratic(G):
    # Create the model
    model = gp.Model()
    model.setParam('OutputFlag', 0)

    # Create the variables
    x = {}
    for i in G.nodes():
        if len(G.nodes[i]['children']) == 0:
            x[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=G.nodes[i]['l'], ub=G.nodes[i]['u'])
    
    n = len(x)
    R = G.nodes[0]['u']

    # Set the objective function
    obj = 0
    leaf_dict = {}
    for i in list(G.nodes)[:0:-1]:
        leaf_dict[i] = [i] if len(G.nodes[i]['children']) == 0 else leaf_dict[G.nodes[i]['children'][0]] + leaf_dict[G.nodes[i]['children'][1]]
        obj -= (G.nodes[i]['weight'] * (n / (R * len(leaf_dict[i])))**2 ) * gp.quicksum((x[j] - R/n) for j in leaf_dict[i])
        obj += (G.nodes[i]['weight'] * (n / (R * len(leaf_dict[i])))**3 ) * gp.quicksum((x[j] - R/n) * (x[k] - R/n) for j, k in itertools.product(leaf_dict[i], leaf_dict[i]))
    leaf_dict[0] = leaf_dict[G.nodes[0]['children'][0]] + leaf_dict[G.nodes[0]['children'][1]]

    model.setObjective(obj, GRB.MINIMIZE)

    # Add the constraints
    for i in list(G.nodes)[::-1]:
        if len(G.nodes[i]['children']) != 0:
            model.addConstr(gp.quicksum(x[j] for j in leaf_dict[i]) >= G.nodes[i]['l'])
            model.addConstr(gp.quicksum(x[j] for j in leaf_dict[i]) <= G.nodes[i]['u'])

    # Optimize the model
    model.optimize()

    return np.array([x[i].x for i in sorted(x.keys())]), model.Runtime
