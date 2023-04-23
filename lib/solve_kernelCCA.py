import numpy as np
from .grammatrices import centered_gram_matrics
import torch


def func_rk(K_i, k):
    # rk(Ki)
    N = K_i.shape[0]
    I = torch.eye(N)
    return (torch.inverse(K_i+N*k*I/2))@K_i


def matrice_Kkhat(estimated_source, k, kernel_type):
    # m:number of sources(x), N:number of samples
    m = estimated_source.shape[0]
    N = estimated_source.shape[1]

    # compute K_i
    Kis = torch.zeros((m, N, N))
    for i in range(m):
        Kis[i] = centered_gram_matrics(estimated_source[i], kernel_type)

    Kk_hat = torch.zeros((m*N, m*N))
    I = torch.eye(N)

    # compute Kk_hat
    for i in range(m):
        for j in range(i+1):
            if i == j:
                Kk_hat[i*N:(i+1)*N, j*N:(j+1)*N] = I
            else:
                Kk_hat[i*N:(i+1)*N, j*N:(j+1)*N] = func_rk(Kis[i], k)@func_rk(Kis[j], k)
    for i in range(m):
        for j in range(i+1, m):
            Kk_hat[i*N:(i+1)*N, j*N:(j+1)*N] = Kk_hat[j*N:(j+1)*N, i*N:(i+1)*N]

    return Kk_hat


def eigenvalues(Kk_hat):
    # solve the eigenval problem: Kk_hat@beta = lamda * beta
    lamda, _ = torch.linalg.eig(Kk_hat)
    return lamda.real





# x = np.arange(1, 13).reshape(4, 3)
# Kk_hat = matrice_Kkhat(x, 0.02, 'Gaussian')
# print(Kk_hat.shape)

