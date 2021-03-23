import numpy as np
import time
import optimization as optim
from numba import roc


def create_batch(x, y, n=100, keep_remainder=True):
    '''Batch generator
    Params:
     - x : input matrix
     - y : target matrix
     - n : batch size
     - keep_remainder (boolean): keep or drop remain if exist
    Output:
     - matrix (batch_size, :)
    '''
    mask = np.random.permutation(len(x))
    x = x[mask]
    y = y[mask]
    if keep_remainder and (len(x) % n != 0):
        num_batch = len(x) // n + 1
    else:
        num_batch = len(x) // n
    for i in range(num_batch):
        yield x[n*i:n*(i+1)], y[n*i:n*(i+1)], num_batch


def update(x, y, name, batch_size, lr_rate, keep_remainder=True, debug=None):
    '''Update function

    Params:
     - x (ndarray) : input matrix in format (num_of_examples, features)
     - y (ndarray) : output target in format (target_of_example, 1)
     - name : choose algorithm used to optim :
            + line_search : Backtracking Line search
            + sgd : Stochastic Gradient Descent
            + momentum_sgd : Stochastic Gradient Descent with momentum
            + adagrad : AdaGrad
            + rmsprop : RMSProp
            + adam : Adam
     - batch_size (scalar): batch size
     - lr_rate (scalar) : learning rate value
     - keep_remainder (scalar): keep or drop remain
     - debug (boolean) : show loss while training if True

    Outputs:
     - w: (ndarray) : optimized hyperparams
     - loss_history (list) : all of losses in training phase
    '''
    if name not in ['line_search', 'sgd', 'momentum_sgd', 'adagrad', 'rmsprop', 'adam']:
        raise AssertionError("Algorithm must be one of ('line_search, 'sgd', 'momentum_sgd', 'adagrad', 'rmsprop', 'adam')")
    
    w = np.random.rand(x.shape[1], 1)
    config = {'lr_rate': lr_rate}
    loss_history = []
    id_epoch = 1
    running = True
    while running:
        start = time.time()
        id_batch = 0
        for bx, by, num_batch in create_batch(x, y, batch_size, keep_remainder):
            dw = bx.T.dot(bx.dot(w) - by) / len(bx)
            loss = np.mean(np.square(bx.dot(w) - by))
            if name == 'line_search':
                w, _ = optim.line_search(bx, by, w, dw, config)
            else:
                w, v = getattr(optim, name)(w, dw, config)
            if loss_history and np.abs(loss_history[-1] - loss) < 1e-5:
                running = False
                if debug:
                    print("Loss before : {} ----- Loss after : {}".format(loss_history[-1], loss))
                print("Converge after {:.4f}s at epoch {} --- batch {}/{} --- loss (training) {}".format(time.time() - start, id_epoch, id_batch, num_batch, loss))
                break
    
            id_batch += 1
            loss_history.append(loss)
        
        if debug and (id_epoch % 20 == 0):
            print("Loss at epoch {} : {}".format(id_epoch, loss_history[-1]))

        id_epoch += 1
    
    return w, loss_history

def update_backtrack(x, y, alpha, beta, debug=None):
    '''Update function for backtrack (plot)
    '''
    w = np.random.rand(x.shape[1], 1)
    config = {'alpha': alpha,
              'beta': beta}
    loss_history = []
    id_epoch = 1
    running = True
    while running:
        dw = x.T.dot(x.dot(w) - y) / len(x)
        loss = np.mean(np.square(x.dot(w) - y))
        w, _ = optim.line_search(x, y, w, dw, config)
        if loss_history and np.abs(loss_history[-1] - loss) < 1e-6:
            running = False
            if debug:
                print("Loss before : {} ----- Loss after : {}".format(loss_history[-1], loss))
            # print("Converge at epoch {} ---- Loss : {}".format(id_epoch, loss))
            break
    
        loss_history.append(loss)
        
        if debug and (id_epoch % 20 == 0):
            print("Loss at epoch {} : {}".format(id_epoch, loss_history[-1]))

        id_epoch += 1
    
    return w, loss_history


def update_gd(x, y, lr_rate, debug=None):
    '''Update function for gd (plot)
    '''
    w = np.random.rand(x.shape[1], 1)
    config = {'lr_rate': lr_rate}
    loss_history = []
    id_epoch = 1
    running = True
    while running:
        dw = x.T.dot(x.dot(w) - y) / len(x)
        loss = np.mean(np.square(x.dot(w) - y))
        w, _ = optim.sgd(w, dw, config)
        if loss_history and np.abs(loss_history[-1] - loss) < 1e-6:
            running = False
            if debug:
                print("Loss before : {} ----- Loss after : {}".format(loss_history[-1], loss))
            # print("Converge at epoch {} ---- Loss : {}".format(id_epoch, loss))
            break
    
        loss_history.append(loss)
        
        if debug and (id_epoch % 20 == 0):
            print("Loss at epoch {} : {}".format(id_epoch, loss_history[-1]))

        id_epoch += 1
    
    return w, loss_history