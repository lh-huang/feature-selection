import scipy
from numpy.linalg import svd
import skfuzzy as fuzz
import math
from utils import *
def suggest_p(n, d, c):
    if n > 1000:p = 0.2  
    elif n > 150:p = 0.4  
    else:p = 0.8 
    if d > 2000: p -= 0.1  
    elif d > 3000: p -= 0.2
    if c > 10: p += 0.1  
    if c > 15:p += 0.2
    p = max(0.2, min(p, 0.85))
    return round(p, 2)
def construct_hypergraph(X, W):
    n, d = X.shape
    _1, c = W.shape
    p=suggest_p(n,d,c)
    w = np.mean(np.abs(W), axis=1)  
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_squared = (X[i] - X[j]) ** 2  
            D[i, j] = np.sum(w * distance_squared)  
    avg_distances = np.mean(D, axis=1)
    threshold = avg_distances * p
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if D[i, j] <= threshold[i]:
                H[i, j] = 1
    hyperedge_degrees = np.sum(H, axis=0)  
    D_e = np.diag(hyperedge_degrees)
    vertex_degrees = np.sum(H, axis=1)  
    D_v = np.diag(vertex_degrees)
    D_e_inv = np.linalg.inv(D_e)  
    H_transpose = H.T
    L = D_v - H @ D_e_inv @ H_transpose
    return L
def cm_init(X, n_samples, n_clusters):
    ct, Y, p0, d, p1, p2, p3 = fuzz.cluster.cmeans(X.transpose(), n_clusters, 1.0005, error=0.0001, maxiter=300, init=None)
    Y = np.transpose(Y)
    G = np.zeros((n_samples, n_clusters), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_clusters):
            G[i, j] = (Y[i, j] / np.sqrt(np.sum(Y[:, j])))
    return G
def fwhsr(X,c, lambda1,lambda2,lambda3,lambda4):
    XX = np.dot(X, np.transpose(X))
    XTX = np.dot(np.transpose(X), X)
    lambda5 = 1e+6
    n, d = X.shape
    W = np.random.rand(d, c)*0.01
    G = cm_init(X, n, c)
    GGT = np.dot(G, np.transpose(G))
    GTG = np.dot(np.transpose(G), G)
    D = np.identity(d)
    one = np.ones((n, n), dtype=np.float64)
    maxIter = 100
    obj = np.zeros(maxIter)
    eps=2e-10
    for iter_step in range(maxIter):
        num1= np.dot(np.transpose(X), G) + lambda4*W
        den1=np.dot(XTX,W) + lambda3*np.dot(D,W) + lambda4 * np.dot(np.ones((d,d)),W)
        T1 = np.divide(num1, den1)
        W = W * np.array(T1)
        L_H = construct_hypergraph(X, W)
        temp = np.sqrt((W*W).sum(1))
        temp[temp < 1e-16] = 1e-16
        temp = 0.5 / temp
        D = np.diag(temp)
        M = np.dot(lambda1*XX + n * lambda5 * one, G)
        num2 =2 * M +  np.dot(X, W)
        den2 = np.dot(M, GTG) + np.dot(GGT, M) + G + lambda2*np.dot(L_H , G)
        T2 = np.divide(num2, den2)
        G = G * np.array(T2)
        GGT = np.dot(G, np.transpose(G))
        GTG = np.dot(np.transpose(G), G)
        GGone = np.dot(GGT, one)
        obj[iter_step] = np.linalg.norm(np.dot(X, W) - G, 'fro')**2 + \
                         lambda1 * np.linalg.norm(X - np.dot(GGT, X), 'fro') ** 2 + \
                         lambda2 * np.trace(np.dot(np.dot(np.transpose(G), L_H), G)) + \
                         lambda3 * (np.sqrt((W*W).sum(1))).sum() + \
                         lambda4 * (np.sum(np.abs(np.dot(W, np.transpose(W)))) - np.linalg.norm(W, 'fro')**2) + \
                         lambda5 * np.linalg.norm(GGone-one, 'fro')**2
        print(obj[iter_step])
        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step-1])/obj[iter_step]  <= 1e-5:
            break
    return W
