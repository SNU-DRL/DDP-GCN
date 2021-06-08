# @Time     : Jan. 10, 2019 15:21
# @Author   : Veritas YIN
# @FileName : math_graph.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs


def scaled_laplacian(W, lambda_max=2):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    n, d = np.shape(W)[0], np.array(W.sum(1))

    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.identity(n) - W.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    if lambda_max is None:
        lambda_max, _ = eigs(L, k=1, which='LR')
        lambda_max = lambda_max[0]
    M, _ = L.shape
    I = np.identity(M, dtype=L.dtype)
    L = (2 / lambda_max * L) - I

    return L.astype(np.float32)



def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError("ERROR: the size of spatial kernel must be greater than 1, but received {}".format(Ks))


def first_approx(W, n, symmetric=True):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    A = W + np.identity(n)
    d = np.sum(A, axis=1)

    if np.sum(np.isnan(d)) == W.shape[0]:
        return np.identity(n)
    elif symmetric:
        sinvD = np.sqrt(np.mat(np.diag(d)).I)
        # refer to Eq.5
        return np.mat(np.identity(n) + sinvD * A * sinvD)
    else:
        sinvD = np.mat(np.diag(d)).I
        return np.mat(np.identity(n) + sinvD * A)


def weight_matrix(file_path, sigma, epsilon, scaling=True, transpose=False, linkIdx=[]):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
        if len(linkIdx) > 0:
            W = W[linkIdx][:, linkIdx]
    except FileNotFoundError:
        print('ERROR: input file was not found in {}.'.format(file_path))

    if transpose:
        W = W.T

    assert np.sum(np.isnan(W)) == 0
    assert np.sum(np.isinf(W)) < W.shape[0] * W.shape[1]
    # print("RAW SPARSITY: %i" %np.sum(W == 0))
    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if sigma == 'var':
        sigma = W[~np.isinf(W)].flatten().var()
    # print(sigma)

    if scaling:
        n = W.shape[0]
        # W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        result = np.exp(-W2 / sigma)
        # print("APPLIED SIGMA: %i" % np.sum(result == 0))
        return result * (result >= epsilon) * W_mask
    else:
        return W
