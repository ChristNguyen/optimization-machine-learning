import numpy as np


def mse_loss(x, y, w):
    '''
    '''
    pred_y = x.dot(w)

    return np.mean(np.square(pred_y - y))

def r2_scoring(x, y, w):
    '''
    '''
    pred_y = x.dot(w)
    mean_y = np.mean(y)
    f = lambda y1, y2: np.sum(np.square(y1 - y2))

    return 1 - f(y, pred_y) / f(y, mean_y)

def adjusted_r2(x, y, w):
    '''
    '''
    rs2 = r2_scoring(x, y, w)
    N = len(x)

    return 1 - ((1-rs2)*(N-1) / (N-5-1))