import networkx as nx

def param_to_data(obj, d, Xv, Xw, a, b):
    n = len(d)
    lbNested, ubNested = [0] * n, [0] * n

    v, w = 0, 0
    for i in range(n):
        v += Xv[i]
        w += Xw[i]
        lbNested[i], ubNested[i] = min(v, w), max(v, w)
    lbNested[n-1] = ubNested[n-1]

    return {
        'dimension': n,
        'objFuncType': obj,
        'lbVar': [1] * n,
        'capacity': d,
        'lbNested': lbNested,
        'ubNested': ubNested,
        'cost_param_a': a,
        'cost_param_b': b,
    }

def data_to_tree(data):
    n = data['dimension']
    G = nx.path_graph(n)
    G.add_nodes_from(range(n, 2*n - 1))

    for i in range(n, 2*n-1):
        G.add_edge(i, 2*n-2-i)

    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        G.nodes[i]['children'] = [x for x in neighbors if x > i]
        G.nodes[i]['parent'] = [x for x in neighbors if x < i]

    # Leaves
    for i in range(n):
        G.nodes[i+n-1]['l'] = data['lbVar'][i]
        G.nodes[i+n-1]['u'] = data['capacity'][i]
        G.nodes[i+n-1]['obj'] = data['objFuncType']
        G.nodes[i+n-1]['param'] = {'a': data['cost_param_a'][i], 'b': data['cost_param_b'][i]}

    # Inner nodes
    for i in range(n-1):
        G.nodes[i]['l'] = data['lbNested'][n-i-1]
        G.nodes[i]['u'] = data['ubNested'][n-i-1]
        G.nodes[i]['obj'] = 'zero'
        G.nodes[i]['param'] = {}

    # Modify the bounds on `x(0)` to reflect bounds for the singleton `{x(0)}`
    G.nodes[n-1]['l'] = max(G.nodes[n-1]['l'], data['lbNested'][0])
    G.nodes[n-1]['u'] = min(G.nodes[n-1]['u'], data['ubNested'][0])

    return G
