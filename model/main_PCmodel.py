"""
David Clark
May 2020
"""

import numpy as np 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

"""
===============================
=== User-defined parameters ===
===============================
"""

PLOT_STUFF = True
SAVE_STUFF = False
FILENAME = "pc_data_output.npz"
N_int = 100 #numer of integrators
N_pc = 400 #number of place cells
alpha = .55 #threshold for place cells
vel_mps = 0.4 #mouse veloicty (m/s)
vel_std_mps = 0.05 #trial-to-trial std. dev. of each integrator's velocity (m/s)
int_noise_std_mps = 0.175 #std. dev. per unit time of integrator drift noise (m/s)
sigma_f_m = 0.35 #smoothness scale of spatial wiggles (m)
track_len_m = 4. #track length (m)
odor_locations_m = np.array([1., 3.]) #locations of the odor cues (m)
padding_m = 8. #track padding length, since noisy integrators can take on values outside of the track (m)
dt_s = 0.050 #numerical integration timestep (s)

trials_per_day = 80 #number of trials per day
num_days = 5 #total number of days (includes odor and no-odor days)
first_odor_day = 1 #first day on which odors are presented (days start at 0)
num_pcs_per_day = 80 #maximum number of place cells that can form each day
new_pc_num = 3 #number of cells that have plateau potentials each trial

"""
==================================
=== Define some other stuff... ===
==================================
"""

if PLOT_STUFF:
	import matplotlib.pyplot as plt

T_s = track_len_m / vel_mps #duration of each trial (s)
t_vals = np.arange(0, T_s, dt_s) #discretized time values for each trial (s)
x_vals = t_vals*vel_mps #discretized position values for each trial (m)
N_t = len(t_vals) #number of discrete time/position steps per trial
odor_idx = (odor_locations_m / (vel_mps * dt_s)).astype(np.int) #discrete time/position index at which odors are presented
num_trials = trials_per_day*num_days #total number of trials accross all days
first_odor_trial = first_odor_day*trials_per_day #index of the first trial on which odors are presented

def gen_spatial_tuning_curves(N_int, x_range, autocorr_scale=.5, n_pts=1000):
    """
    Generates spatial tuning curves (wiggly functions of space).
    N_int -- number of tuning curves to generate
    x_range -- tuple (x_min, x_max) specifying the spatial range (m)
    autocorr_scale -- smoothness scale of wiggles (m)
    n_pts -- number of points to sample for each tuning curve

    Each tuning curve is sampled from a zero-mean Gaussian process with mean zero
    and autocovariance given by
    f(dt) = exp(-dt^2 / (2 * sigma^2)),
    where sigma = autocorr_scale. To do this, we apply Gaussian smoothing to white noise.
    The function F returned by this function maps a length N_int vector to a length N_int vector.
    That is, F({x1, x2, x3}) = {f1(x1), f2(x2), f3(x3)}, where 'fi' is the i-th tuning curve.
    """
    L = x_range[1] - x_range[0]
    sigma = n_pts * autocorr_scale / L
    noise = np.random.normal(0, 1, (n_pts, N_int))
    sigma_prime = sigma / np.sqrt(2)
    noise_smooth = (gaussian_filter1d(noise, sigma=sigma_prime, axis=0) #, mode="wrap")
                * np.sqrt(sigma_prime * np.sqrt(np.pi) * 2))
    x_interp_vals = np.linspace(x_range[0], x_range[1], n_pts)
    F_all = [interp1d(x_interp_vals, noise_smooth[:, i]) for i in range(N_int)]
    def F(x):
        res = np.array([F_all[i](x[i]) for i in range(N_int)])
        return res
    return F

x_range = (-padding_m, track_len_m+padding_m)
F = gen_spatial_tuning_curves(N_int, x_range, sigma_f_m)
W = np.zeros((N_pc, N_int)) #integrator-->cell weights
M = np.zeros((N_int, N_pc)) #cell-->integrator reset weights
place_cells = np.array([], dtype=np.int)

pc_matrix = np.zeros((num_trials, N_pc))
all_rates = np.zeros((num_trials, N_t, N_pc))
all_ints = np.zeros((num_trials, N_t, N_int))
pc_quota = num_pcs_per_day

for trial_idx in range(num_trials):
	print("trial", trial_idx+1, "of", num_trials)
	#'X" holds integrator values
	X = np.zeros((N_t, N_int))
	#randomly initalize the integrators
	X[0] = np.random.normal(0, 0.2, N_int)
	#create storage for other variables
	f_of_x = np.zeros((N_t, N_int))
	f_of_x_normalized = np.zeros((N_t, N_int))
	I = np.zeros((N_t, N_pc))
	R = np.zeros((N_t, N_pc))
	#sample velocities for integrators
	noisy_vel_mps = np.random.normal(vel_mps, vel_std_mps, N_int)
	#run the trial...
	for t_idx in range(0, N_t):
		#perform integration step
		if t_idx > 0:
			X[t_idx] = X[t_idx-1] + dt_s*noisy_vel_mps + np.random.normal(0, int_noise_std_mps*np.sqrt(dt_s), N_int)
		#pass integrator values through tuning curves
		f_of_x[t_idx] = F(X[t_idx])
		#normalize
		f_of_x_normalized[t_idx] = f_of_x[t_idx] / np.linalg.norm(f_of_x[t_idx])
		#get place cell input
		I[t_idx] = np.dot(W, f_of_x_normalized[t_idx])
		#apply noise to each PC's threshold
		alpha_noisy = alpha + np.random.normal(0, alpha*0.1, N_pc)
		#perform thresholding
		R[t_idx] = np.maximum(I[t_idx] - alpha_noisy, 0)
		#reset integrators if necessary
		if trial_idx >= first_odor_trial and t_idx in odor_idx:
			#reset integrators
			if np.sum(R[t_idx] > 1e-12) >= 1:
				X[t_idx] = np.dot(M, R[t_idx]) / np.sum(R[t_idx])

	#if we're at the start of a new day, increase the PC quota
	if trial_idx % trials_per_day == 0 and trial_idx > 0:
		pc_quota = pc_quota + num_pcs_per_day

	#define a few subsets of cells
	active_cells = np.arange(N_pc)[np.sum(R**2, axis=0) > 1e-12]
	place_cells = np.union1d(place_cells, active_cells)
	inactive_cells = np.setdiff1d(np.arange(N_pc), active_cells)
	non_place_cells = np.setdiff1d(np.arange(N_pc), place_cells)
	num_cells_needed = pc_quota - len(place_cells)

	#make sure that your weights are 0 if you're not a place cell
	W[non_place_cells, :] = 0.
	M[:, non_place_cells] = 0.

	if num_cells_needed > 0:
		#implement plateau potentials (plasticity) for 'new_pc_num' randomly selected cells
		plastic_cells = np.random.choice(non_place_cells, size=min(num_cells_needed, new_pc_num), replace=False)
		print(num_cells_needed, "PCs needed")
		place_field_idx = np.random.choice(np.arange(N_t), size=len(plastic_cells), replace=True)
		W[plastic_cells, :] = f_of_x_normalized[place_field_idx]
		M[:, plastic_cells] = X[place_field_idx].T
	else:
		print("Plasticity complete.")
		print(len(place_cells), "place cells")

	if SAVE_STUFF:
		#write the current state of things to file
		pc_matrix[trial_idx, place_cells] = 1.
		all_rates[trial_idx] = R
		all_ints[trial_idx] = X
		np.savez(FILENAME, pc_matrix=pc_matrix, all_rates=all_rates, all_ints=all_ints)

	if PLOT_STUFF:
		#plot stuff
		active_disp_order = np.argsort(np.argmax(R[:, active_cells], axis=0))
		disp_order = np.concatenate((active_cells[active_disp_order], inactive_cells ))
		plt.imshow(R[:, disp_order].T)
		plt.xlabel("distance")
		plt.ylabel("place cell index")
		if trial_idx >= first_odor_trial:
			plt.title("Odors present")
		else:
			plt.title("No odors present")
		plt.show(block=False)
		plt.pause(0.001)
		plt.close()



