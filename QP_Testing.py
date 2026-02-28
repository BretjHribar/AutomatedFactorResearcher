import numpy as np
import matplotlib.pyplot as plt
import time
import operator
import math

import scipy
from scipy.optimize import minimize

fees = 0.002

def get_obj_func(h0, alpha_vec):
    def obj_func(h):
        #f = 0
        #f = 0.5 * risk_aversion * np.sum(np.matmul(Q, h) ** 2)
        f = np.dot(h, alpha_vec)
        errors = h - alpha_vec
        f = np.abs(np.sum(errors**2))
        ########f += np.dot((h - h0) ** 2, Lambda)
        ##f += np.nansum(((h - h0) ** 2) * Lambda)
        return f
    return obj_func

def get_grad_func(h0, alpha_vec):
    def grad_func(h):
        #g = risk_aversion * np.matmul(QT, np.matmul(Q, h))
        #g = np.dot(alpha_vec,-1.0)
        g = -alpha_vec
        #g += 2 * (h - h0) * Lambda
        return np.asarray(g)
    return grad_func
old_positions = np.asarray([0.5,0.3,-0.4])
new_target_pos = np.asarray([-0.5,0.3,0.4])

obj_func = get_obj_func(old_positions, new_target_pos )

obj_func = get_obj_func(old_positions, new_target_pos )
grad_func = get_grad_func(old_positions, new_target_pos)

optimizer_result = scipy.optimize.fmin_l_bfgs_b(obj_func, old_positions, approx_grad=True)
#optimizer_result = scipy.optimize.fmin_l_bfgs_b(obj_func, old_positions, fprime=grad_func)

#optimizer_result = minimize(obj_func, h0, bounds=weightBounds, method='Nelder-Mead', options={'maxiter': 1000})

#optimizer_result = minimize(obj_func, old_positions)
#optimizer_result = minimize(obj_func, old_positions, method='Nelder-Mead', options={'maxiter': 1000})
# optimizer_result = minimize(obj_func, h0, method='Nelder-Mead')

TEST = 1