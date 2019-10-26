import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# setup basic constants and functions
zx = 5
zy = 5

# get label and color attributes
def cattr(N=50):
    color = np.ones(2*N)
    color[0:N] = 0
    carray = ['skyblue', 'lightsalmon']
    cmap = ListedColormap(carray)
    return color, carray, cmap

# projection of points onto base
def proj(x, y, bx, by, zx=0, zy=0):
    w = (x -zx) * bx + (y - zy) * by
    return [w * bx + zx , w * by + zy]

# center of distribution
def center(po):
    cx = np.average(po[:,0])
    cy = np.average(po[:,1])    
    return np.array([cx, cy])

# variance matrix of v1 and v2
def var(v1, v2):
    x = v1 - v2
    return x.reshape([2, 1]) * x


def within(po):
    cen = center(po)
    dis = [var(x, cen) for x in po]
    return np.average(dis)

# uniform
def gen_uni(N=50):
    x1 = np.random.uniform(0,4,N)
    y1 = np.random.uniform(2,8,N)
    x2 = np.random.uniform(3,9,N)
    y2 = np.random.uniform(0.5,3.5,N)
    data = np.concatenate((np.array([x1, y1]).T, 
                          np.array([x2, y2]).T), axis=0)
    return data
    
# cyclic
def gen_cyc(N=50):
    # set data distribution
    mean1 = [2, 5]
    cov1 = [[0.2, 0], [0, 0.2]]
    x1 = np.random.multivariate_normal(mean1, cov1, N)

    mean2 = [6, 2]
    cov2 = [[0.25, 0], [0, 0.25]]
    x2 = np.random.multivariate_normal(mean2, cov2, N)

    data = np.concatenate((x1, x2), axis=0)    
    return data

# skewed normal distribution
def gen_norm(N=50, c1=[[0.1, 0], [0, 0.4]],c2=[[0.3, 0], [0, 0.1]]):
    # set data distribution
    mean1 = [2, 5]
    cov1 = c1
    x1 = np.random.multivariate_normal(mean1, cov1, N)

    mean2 = [6, 2]
    cov2 = c2
    x2 = np.random.multivariate_normal(mean2, cov2, N)

    data = np.concatenate((x1, x2), axis=0)
    
    return data

# use lda to find the projections of data
def lda(X, y, zx=5, zy=5):
    mu0 = np.mean(X[y == 0], axis=0)
    mu1 = np.mean(X[y == 1], axis=0)
    B = var(mu0, mu1)
    s0 = np.zeros([2, 2])
    for xi in X[y == 0]:
        s0 += var(xi, mu0)
    s1 = np.zeros([2, 2])
    for xi in X[y == 1]:
        s1 += var(xi, mu1)
    S = s0 + s1
    S_inv = np.linalg.inv(S)
    S_inv_B = S_inv.dot(B)
    eig_vals, eig_vecs = np.linalg.eig(S_inv_B)
    # sort eigenvalues and eigenvectors
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx] # Not needed
    eig_vecs = eig_vecs[:, idx]
    W = eig_vecs[:, 0]
    pxs = []
    for pt in X:
        pxs.append(((pt - zx) @ W * W + zy).reshape([2, ]).tolist())
    pxs = np.array(pxs)
    return pxs, W, mu0, mu1
