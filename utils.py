import numpy as np

def solve_univariate_piecewise_linear(l1, u1, b1, l2, u2, b2, c):
    # Case 1: b1 <= l1 and b2 <= l2
    if b1 + b2 <= c:
        x1 = max(b1, l1, c - u2)
    # Case 2: b1 >= u1 and b2 >= u2
    elif b1 + b2 >= c:
        x1 = min(b1, u1, c - l2)
    # Case 3: b1 <= l1 and b2 >= u2
    elif b1 - u2 <= c - l2:
        x1 = max(b1, l1, c - u2)
    # Case 4: b1 >= u1 and b2 <= l2
    elif b1 - l2 >= c - u1:
        x1 = min(b1, u1, c - l2)
    # Case 5: b1 lies between l1 and u1, b2 <= l2
    elif b1 + b2 <= l1 + l2:
        x1 = max(b1, c - u2)
    # Case 6: b1 lies between l1 and u1, b2 >= u2
    elif b1 + b2 >= u1 + u2:
        x1 = min(b1, c - l2)
    # Case 7: b2 lies between l2 and u2, b1 <= l1
    elif b1 + b2 <= l1 + l2:
        x1 = max(b1, c - u2)
    # Case 8: b2 lies between l2 and u2, b1 >= u1
    elif b1 + b2 >= u1 + u2:
        x1 = min(b1, c - l2)
    # Case 9: both b1 and b2 lie between l1 and u1/u2
    else:
        x1 = (c - b2 + b1) // 2
    
    x2 = c - x1
    return x1, x2


def projection(y, G):
    leaves = [i for i in G.nodes() if len(G.nodes[i]['children']) == 0]

    gdict = {}
    for i in list(G.nodes())[::-1]:
        if len(G.nodes[i]['children']) == 0:
            g = np.array([G.nodes[i]['l'], y[i - leaves[0]], G.nodes[i]['u']]) 
            gdict[i] = np.clip(g, np.ones_like(g)*G.nodes[i]['l'], np.ones_like(g)*G.nodes[i]['u'])
        else:
            g1, g2 = gdict[G.nodes[i]['children'][0]], gdict[G.nodes[i]['children'][1]]
            gdict[i] = np.clip(g1 + g2, np.ones_like(g1)*G.nodes[i]['l'], np.ones_like(g1)*G.nodes[i]['u'])

    R = G.nodes[0]['u']
    xdict = {0:R}
    for i in G.nodes():
        if len(G.nodes[i]['children']) == 0:
            continue
        l1, b1, u1 = gdict[G.nodes[i]['children'][0]]
        l2, b2, u2 = gdict[G.nodes[i]['children'][1]]
        x1, x2 = solve_univariate_piecewise_linear(l1, u1, b1, l2, u2, b2, xdict[i])
        xdict[G.nodes[i]['children'][0]], xdict[G.nodes[i]['children'][1]] = x1, x2

    return np.array([xdict[i] for i in leaves])

def objective(node, x):
    """
    Objective functions for f, crash, and fuel are retrieved from
    https://github.com/qqqhe/dca/blob/3c80051a1062e1e00691615d9c469fe24f5a2ee0/src/main/java/dca_ijoc/RAPNCTestUtils.java#L48-L87
    """

    if not node['l'] <= x <= node['u']:
        return np.inf
    elif node['obj'] == 'staff':
        w = node['param']['w']
        return w / x if x > 0 else np.inf
    elif node['obj'] == 'f':
        b = node['param']['b']
        return x**4 / 4 + b*x
    elif node['obj'] == 'crash':
        a, b = node['param']['a'], node['param']['b']
        return 10*b + a/x if x > 0 else np.inf
    elif node['obj'] == 'fuel':
        a, b = node['param']['a'], node['param']['b']
        return a * b**2 / x**3 if x > 0 else np.inf
    elif node['obj'] == 'zero':
        return 0
    else:
        assert False, 'Unknown objective function type'

def direction(x, G):
    leaves = [i for i in G.nodes() if len(G.nodes[i]['children']) == 0]

    xdict = {}
    for i in list(G.nodes())[::-1]:
        if len(G.nodes[i]['children']) == 0:
            xdict[i] = x[i-leaves[0]]
        else:
            xdict[i] = xdict[G.nodes[i]['children'][0]] + xdict[G.nodes[i]['children'][1]]

    wdict = {}
    for i in list(G.nodes())[:0:-1]:
        p = G.nodes[i]['parent'][0]
        wdict[(i, p)] = objective(G.nodes[i], xdict[i]-1) - objective(G.nodes[i], xdict[i])
        wdict[(p, i)] = objective(G.nodes[i], xdict[i]+1) - objective(G.nodes[i], xdict[i])
    
    Lu, Ld, Ls = {}, {}, {}
    Pu, Pd, Ps = {}, {}, {}
    for i in list(G.nodes())[::-1]:
        if len(G.nodes[i]['children']) == 0:
            Lu[i], Ld[i], Ls[i] = 0, 0, 0
            Pu[i], Pd[i], Ps[i]  = i, i, (i, i)
        else:
            c1, c2 = G.nodes[i]['children']
            Lu[i] = min(Lu[c1] + wdict[(c1, i)], Lu[c2] + wdict[(c2, i)])
            Pu[i] = Pu[c1] if Lu[c1] + wdict[(c1, i)] <= Lu[c2] + wdict[(c2, i)] else Pu[c2]
            Ld[i] = min(Ld[c1] + wdict[(i, c1)], Ld[c2] + wdict[(i, c2)])
            Pd[i] = Pd[c1] if Ld[c1] + wdict[(i, c1)] <= Ld[c2] + wdict[(i, c2)] else Pd[c2]
            Ls[i] = min([Ls[c1], Ls[c2], Lu[i]+Ld[i]])
            if Ls[c1] == Ls[i]:
                Ps[i] = Ps[c1]
            elif Ls[c2] == Ls[i]:
                Ps[i] = Ps[c2]
            else:
                Ps[i] = (Pu[i], Pd[i])

    return Ls[0], (Ps[0][0] - leaves[0], Ps[0][1] - leaves[0])


def greedy(pred, G):
    x = projection(np.round(pred), G)
    itr = 0
    while True:
        itr += 1
        dec, (i, j) = direction(x, G)
        if dec > 0:
            break
        x[i] -= 1
        x[j] += 1
    return itr, x


# based on https://www.researchgate.net/publication/343831904_NumPy_SciPy_Recipes_for_Data_Science_Projections_onto_the_Standard_Simplex#pf1
def simplex_projection(vecX, scale):
    m = vecX.size
    vecS = np.sort(vecX)[::-1]
    vecC = np.cumsum(vecS) - scale
    vecH = vecS - vecC / (np.arange(m) + 1)
    vecH[vecH<=0] = np.inf
    r = np.argmin(vecH)
    t = vecC[r] / (r + 1)
    vecY = vecX - t
    vecY[vecY<0] = 0
    return vecY

def integer_initial_vector(n, R):
    """
    Returns an initial vector of length n with integer values that sum to R.
    """

    x = np.ones(n) * (R // n)
    indices = np.random.choice(n, R % n, replace=False)
    x[indices] += 1

    return x
