"""
David Clark
January 2020
"""

import numpy as np 
import h5py
import os
from math import floor
import numpy.ma as ma
from scipy.stats import sem
import warnings
from scipy.stats import linregress
import data_util

#x_trials scale = 0,...,4.
def compute_pred_slope(x_trials):
    N_trials = len(x_trials)
    N_t = np.max([len(x) for x in x_trials])
    all_x = np.ones((N_trials, N_t)) * 4.
    for i in range(N_trials):
        x = x_trials[i]
        all_x[i, :len(x)] =  x
    x_avg = all_x.mean(axis=0)

    vel_vals = np.array([4. / (len(x) * .05) for x in x_trials])
    vel_avg = np.mean(vel_vals)
    T_avg = 4. / vel_avg
    
    T_avg_idx = int(np.round(T_avg / .05))
    slope = -x_avg[T_avg_idx] + np.mean(x_avg[:T_avg_idx])
    
    return slope / vel_avg

#x_trials scale = 0,...,4.
def compute_pred_slope_smart(x_trials):
    x_trials = [x[x > 0.1] for x in x_trials]
    N_trials = len(x_trials)
    N_t = np.max([len(x) for x in x_trials])
    all_x = np.ones((N_trials, N_t)) * 4.
    for i in range(N_trials):
        x = x_trials[i]
        all_x[i, :len(x)] =  x
    x_avg = all_x.mean(axis=0)
    vel_vals = np.array([3.9 / (len(x) * .05) for x in x_trials])
    vel_avg = np.mean(vel_vals)
    vel_deviations = vel_vals - vel_avg
    err_vals = np.array([ np.mean(x_avg[:len(x_trials[i])] - x_trials[i]) for i in range(len(x_trials)) ])
    slope = linregress(vel_deviations, err_vals)[0]
    return slope

def compute_pred_slope_for_session(mouse_num, day_num):
    data = data_util.load_single_trial_data(mouse_num, day_num, stop_pad=0)
    x_trials = data["x"][0]
    if day_num in (1, 2, 3, 4):
        x_trials = x_trials + data["x"][1]
    x_trials = [x*1e-3 for x in x_trials]
    return compute_pred_slope_smart(x_trials)

def compute_pred_slope_all():
    slopes_mice_days = np.zeros((5, 6))
    for mouse_num in range(1, 6):
        for day_num in range(6):
            slopes_mice_days[mouse_num-1, day_num] = compute_pred_slope_for_session(mouse_num, day_num)
    return slopes_mice_days
    


