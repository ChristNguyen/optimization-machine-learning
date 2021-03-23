import numpy as np


# Closed form
def normal_equation(x, y):
    '''OLS Solver
    '''
    w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    return w

# First order
def line_search(x, y, w, dw, config):
    '''Backtracking Line search
    '''
    def mse_loss(x, y, w):
        return np.mean(np.square(x.dot(w) - y))
    
    if config is None:
        config = {}
    config.setdefault('alpha', .5)
    config.setdefault('beta', .5)
    t = 1
    while mse_loss(x, y, w-t*dw) > (mse_loss(x, y, w)- config['alpha']*t*np.linalg.norm(dw)**2):
        t = t*config['beta']
    w -= t*dw
    
    return w, config
    
def sgd(w, dw, config=None):
    '''Vanilla SGD
    '''
    if config is None:
        config = {}
    config.setdefault('lr_rate', 1e-2)
    w -= config['lr_rate'] * dw
    return w, config

def momentum_sgd(w, dw, config=None):
    '''
    Momentum SGD
    Params :
     - lr_rate
     - momentum(0, 1) : setting momentum = 0 reduces to vanilla sgd
     - velocity: same shape as w & dw to store moving average of the gradient
    '''
    if config is None:
        config = {}
    config.setdefault('lr_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    v = config['momentum'] * v + config['lr_rate'] * dw
    next_w = w - v
    config['velocity'] = v

    return next_w, v

def adagrad(w, dw, config=None):
    '''
    Adagrad optimization
    Params:
     - lr_rate
     - decay_rate
     - epsilon
     - cache
    '''
    if config is None:
        config = {}
    config.setdefault('lr_rate', 1e-5)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))
    next_w = None

    config['cache'] += dw ** 2
    next_w = w - config['lr_rate'] * dw / (np.sqrt(config['cache']) + config['epsilon'])

    return next_w, config

def rmsprop(w, dw, config=None):
    '''
    Params:
     - lr_rate : scalar learning rate
     - decay_rate
     - epsilon
     - cache
    '''
    if config is None:
        config = {}
    config.setdefault('lr_rate', 1e-5)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))
    next_w = None

    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw ** 2
    next_w = w - config['lr_rate'] * dw / (np.sqrt(config['cache'] + config['epsilon']))

    return next_w, config


def adam(w, dw, config=None):
    '''
    Performing Adam Optimizer
    Params:
     - lr_rate: Scalar learning rate.
     - beta1: Decay rate for moving average of first moment of gradient.
     - beta2: Decay rate for moving average of second moment of gradient.
     - epsilon: Small scalar used for smoothing to avoid dividing by zero.
     - m: Moving average of gradient (momentum)
     - v: Moving average of squared gradient (uncentered)
     - t: Iteration number
    '''
    if config is None:
        config = {}
    config.setdefault('lr_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    # for i in range(config['t']):
    t = config['t'] + 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dw ** 2
    m_hat = config['m'] / (1 - np.power(config['beta1'], t))
    v_hat = config['v'] / (1 - np.power(config['beta2'], t))
    next_w = w - config['lr_rate'] * m_hat / (np.sqrt(v_hat) + config['epsilon'])

    return next_w, config


# Second order
# def newtow():

# def bfgs():

# def conjugate_grad():
