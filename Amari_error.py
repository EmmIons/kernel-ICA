import torch
import numpy as np

if __name__ == '__main__':
    trueW = torch.load('./weights/trueW.pt').cuda()
    estimatedW = torch.load('./weights/estimatedW.pt')
    estimatedW = estimatedW.to(torch.float64).cuda()
    print(estimatedW)
    print(trueW)
    # compute amari error
    trueW_inverse = torch.linalg.inv(trueW)
    m = trueW.size()[0]
    a = estimatedW @ trueW_inverse
    a = torch.abs(a)
    sum_1 = 0
    for i in range(m):
        maxj = torch.max(a[i, :])
        sum_tmp = 0
        for j in range(m):
            sum_tmp += a[i][j]
        sum_1 += (sum_tmp/maxj) - 1

    sum_2 = 0
    for j in range(m):
        maxi = torch.max(a[:, j])
        sum_tmp = 0
        for i in range(m):
            sum_tmp += a[i][j]
        sum_2 += (sum_tmp/maxi) - 1

    print('Amari error:', (sum_1+sum_2)/(2*m))
