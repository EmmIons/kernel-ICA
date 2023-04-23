import numpy as np
import torch


def gaussian_kernel_func(x, y, sigma):
    return torch.exp(-(x-y)**2/2*(sigma**2))


def centered_gram_matrics(x, kernel_type):
    # x:one source of sources(x_i)
    if kernel_type == 'Gaussian':
        kernel_func = gaussian_kernel_func
    N = x.shape[0]
    L = torch.zeros((N, N))
    for i in range(N):
        for j in range(i+1):
            # sigma = 1
            L[i][j] = kernel_func(x[i], x[j], 1)
    for i in range(N):
        for j in range(i+1, N):
            L[i][j] = L[j][i]
    I = torch.eye(N)
    ones = torch.ones(N)
    P = I - ones/N
    # return K_i
    return P@L@P
