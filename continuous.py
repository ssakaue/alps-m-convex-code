import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import itertools

def diff_1st(node, x):
    if node['obj'] == 'staff':
        w = node['param']['w']
        return -w / x**2 if x != 0 else np.nan
    elif node['obj'] == 'f':
        b = node['param']['b']
        return x**3 + b
    elif node['obj'] == 'crash':
        a, b = node['param']['a'], node['param']['b']
        return -a / x**2 if x != 0 else np.nan
    elif node['obj'] == 'fuel':
        a, b = node['param']['a'], node['param']['b']
        return -3 * a * b**2 / x**4 if x != 0 else np.nan
    elif node['obj'] == 'zero':
        return 0
    else:
        assert False, 'Unknown objective function type'

def diff_2nd(node, x):
    if node['obj'] == 'staff':
        w = node['param']['w']
        return 2 * w / x**3 if x != 0 else np.nan
    elif node['obj'] == 'f':
        b = node['param']['b']
        return 3 * x**2
    elif node['obj'] == 'crash':
        a, b = node['param']['a'], node['param']['b']
        return 2 * a / x**3 if x != 0 else np.nan
    elif node['obj'] == 'fuel':
        a, b = node['param']['a'], node['param']['b']
        return 12 * a * b**2 / x**5 if x != 0 else np.nan
    elif node['obj'] == 'zero':
        return 0
    else:
        assert False, 'Unknown objective function type'

def solve_continuous_quadratic(G, yinit):
    # Create the model
    model = gp.Model()
    model.setParam('OutputFlag', 0)

    # Create the variables
    x = {}
    y = {}
    k = 0
    for i in G.nodes():
        if len(G.nodes[i]['children']) == 0:
            x[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=G.nodes[i]['l'], ub=G.nodes[i]['u'])
            y[i] = yinit[k]
            k += 1

    leaf_dict = {}
    for i in list(G.nodes)[:0:-1]:
        leaf_dict[i] = [i] if len(G.nodes[i]['children']) == 0 else leaf_dict[G.nodes[i]['children'][0]] + leaf_dict[G.nodes[i]['children'][1]]
    leaf_dict[0] = leaf_dict[G.nodes[0]['children'][0]] + leaf_dict[G.nodes[0]['children'][1]]

    obj = 0
    for i in G.nodes:
        if G.nodes[i]['obj'] != 'zero':
            obj += diff_1st(G.nodes[i], sum(y[j] for j in leaf_dict[i])) * gp.quicksum((x[j] - y[j]) for j in leaf_dict[i])
            obj += 0.5 * diff_2nd(G.nodes[i], sum(y[j] for j in leaf_dict[i])) * gp.quicksum((x[j] - y[j]) * (x[k] - y[k]) for j, k in itertools.product(leaf_dict[i], leaf_dict[i]))

    model.setObjective(obj, GRB.MINIMIZE)

    # Add the constraints
    for i in list(G.nodes)[::-1]:
        if len(G.nodes[i]['children']) != 0:
            model.addConstr(gp.quicksum(x[j] for j in leaf_dict[i]) >= G.nodes[i]['l'])
            model.addConstr(gp.quicksum(x[j] for j in leaf_dict[i]) <= G.nodes[i]['u'])

    # Optimize the model
    model.optimize()

    return np.array([x[i].x for i in sorted(x.keys())]), model.Runtime
