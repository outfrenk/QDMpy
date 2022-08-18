import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numba import jit


@jit
def model_vector(x, parameter, modelID):
    out = []
    model = [None, ESRSINGLE, ESR15N, ESR14N][modelID]
    for p in parameter:
        out.append(model(x, p))
    return np.array(out)


def ESR14N(x, parameter):
    """
    ESR14N model
    """
    out = []
    AHYP = 0.002158

    for i in range(parameter.shape[0]):
        p = parameter[i]
        aux1 = x - p[0] + AHYP
        width_squared = p[1] * p[1]

        dip1 = p[2] * width_squared / (aux1 * aux1 + p[1] * p[1])

        aux2 = x - p[0]
        dip2 = p[3] * width_squared / (aux2 * aux2 + p[1] * p[1])

        aux3 = x - p[0] - AHYP
        dip3 = p[4] * width_squared / (aux3 * aux3 + p[1] * p[1])

        out.append(1 + p[5] - dip1 - dip2 - dip3)
    return out


def ESR15N(x, parameter):
    """
    ESR15N model
    """
    out = []
    AHYP = 0.0015

    for i in range(parameter.shape[0]):
        p = parameter[i]
        width_squared = p[1] * p[1]

        aux1 = x - p[0] + AHYP
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)

        aux2 = x - p[0] - AHYP
        dip2 = p[3] * width_squared / (aux2 * aux2 + width_squared)

        out.append(1 + p[4] - dip1 - dip2)
    return out


def ESRSINGLE(x, parameter):
    """
    ESR15N model
    """
    out = []

    for i in range(parameter.shape[0]):
        p = parameter[i]
        width_squared = p[1] * p[1]

        aux1 = x - p[0]
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)

        out.append(1 + p[3] - dip1)
    return out


if __name__ == '__main__':
    p_ = np.load(r'C:\Users\micha\Desktop\CAGEO_data\fitdata.npz')
    p = p_['arr_0'][0, 0]
    x = np.linspace(2.8, 2.9, 50)
    print(x.shape, p.shape)
    print(ESR14N(x, p).shape)
