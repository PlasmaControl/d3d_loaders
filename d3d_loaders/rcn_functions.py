import numpy as np
import scipy.signal
import scipy.sparse


def rcn_infer(w_in,w_res,w_bi,w_out,leak,r_prev,u):
    if scipy.sparse.issparse(w_in): # applying input weights to the input. Sparse and dense matrix multiplication is different in Python 
        a1 = w_in * u 
    else:
        a1=np.dot(w_in, u)
    a2 = w_res * r_prev # applying recurrent weights to the previous reservoir states
    r_now = np.tanh(a1 + a2 + w_bi) # adding bias and applying activation function
    r_now = (1 - leak) * r_prev + leak * r_now # applying leak rate
    y = np.dot(np.append([1],r_now),w_out) # applying the output weight
    return r_now,y


