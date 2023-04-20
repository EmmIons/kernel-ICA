import numpy as np
from lib import solve_kernelCCA

import torch
import torch.nn as nn
from torch.autograd import Variable

from scipy.linalg import sqrtm


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, y):
        # X_T = Y_T@W_T, here params of the Linear is equal to W_T
        # output is still X, not X_T
        X_T = self.fc(y.T)
        return X_T.T


if __name__ == '__main__':
    N = 100  # sample size
    m = 4  # number of sources

    # generate X ,size: N*m
    X = np.random.uniform(0, 1, size=(m, N))
    # given A
    A = np.tri(m, m, 0)
    # Y_hat = AX
    Y_hat = A @ X

    # mean-centering
    row_means = np.mean(Y_hat, axis=1)
    Y_centered = Y_hat - row_means[:, np.newaxis]

    # whiten Y
    cov = np.cov(Y_hat, rowvar=True)
    U, S, V = np.linalg.svd(cov)
    d = np.diag(1.0 / np.sqrt(S))
    whiteM = np.dot(U, np.dot(d, U.T))
    Y = np.dot(whiteM, Y_centered)

    # save true W
    trueW = np.linalg.inv(whiteM @ A)
    trueW = torch.from_numpy(trueW)
    torch.save(trueW, './weights/trueW.pt')

    # train model
    model = Model(m, m).cuda()    # model:input:Y  output:X,the params of model is equal to W_T
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epoch = 0
    # load data
    Y = torch.from_numpy(Y).to(torch.float32).cuda()
    # training
    for epoch in range(100):
        model.train()
        Y = Variable(Y)
        X_estimated = model(Y)
        Kk_hat = solve_kernelCCA.matrice_Kkhat(X_estimated, 0.03, 'Gaussian')
        lamda = solve_kernelCCA.eigenvalues(Kk_hat)
        lamda_hat = torch.min(lamda)
        cw = -0.5*torch.log10(lamda_hat)
        cw = lamda_hat
        optimizer.zero_grad()
        cw.backward()
        optimizer.step()
        # show the change of W
        for name, parms in model.named_parameters():
            print('-->name:', name)
            print('-->para:', parms)
            print('-->grad_requirs:', parms.requires_grad)
            print('-->grad_value:', parms.grad)
            print("===")
        print('Epoch [{}], C(W): {:.4f}'.format(epoch+1, cw.item()))

    # save estimated W
    torch.save(model.state_dict(), './weights/estimatedW.pt')