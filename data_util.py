"""
David Clark
December 2019
"""

import numpy as np 
import h5py
import os
from math import floor
import warnings
from scipy.stats import linregress, ranksums, sem


IND_MICE_DIR = "../OlfactoryLandmarksNeuronData/Spatially_binned/"
PRETRAINING_DATA_DIR = "../OlfactoryLandmarksNeuronData/Single_trial/pretraining_behavior_data/"
RAW_IMAGING_DATA_DIR = "../OlfactoryLandmarksNeuronData/Single_trial/Raw_Imaging_data/"
MOUSE_NAMES = ["wfC318", "wfC321", "wfC323", "wfC406", "wfC409"]

def sigfigs(X, precision=2):
    X_sigfigs = np.zeros_like(X)
    for index, x in np.ndenumerate(X):
        X_sigfigs[index] = float(np.format_float_scientific(x, precision=precision))
    return X_sigfigs

def nansem(X, axis=None):
    if axis is None:
        axis = range(X.ndim)
    return np.nanstd(X, axis=axis) / np.sqrt(np.sum(~np.isnan(X), axis=axis))

def ranksums_nan(x, y):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    return ranksums(x, y)

def perform_shuffle(X, X_z, roll=False, recenter=None):
    if len(X) == 0:
        return [], []
    X_shuffled = np.zeros_like(X)
    X_z_shuffled = np.zeros_like(X_z)
    N_trials, N_bins, N_nrns = X.shape
    if recenter is not None:
        centers = np.argmax(X.mean(axis=0)[1:], axis=0) + 1
        for i in range(X.shape[-1]):
            roll_amt = recenter - centers[i]
            X_shuffled[:,:,i] = np.roll(X[:,:,i], roll_amt)
            X_z_shuffled[:,:,i] = np.roll(X_z[:,:,i], roll_amt)
    else:
        for trial_idx in range(N_trials):
            if roll:
                perm = np.roll(np.arange(N_bins), np.random.choice(np.arange(N_bins)))
            else:
                perm = np.random.permutation(N_bins)
            X_shuffled[trial_idx, :, :] = X[trial_idx, perm, :]
            X_z_shuffled[trial_idx, :, :] = X_z[trial_idx, perm, :]
    return X_shuffled, X_z_shuffled

def get_pc_count_from_data(X_1, X_z_1, X_2=None, X_z_2=None):
    pc_idx = get_pc_idx(X_1, X_z_1)
    if X_2 is not None:
        pc_idx_2 = get_pc_idx(X_2, X_z_2)
        pc_idx = np.union1d(pc_idx, pc_idx_2)
    return len(pc_idx)

def get_pc_idx_for_session(mouse_num, day_num, trial_type=0):
    data = load_preprocessed_data(mouse_num, day_num)
    X, X_z = data["neural"], data["z_score"]
    pc_idx = get_pc_idx(X[trial_type], X_z[trial_type])
    return pc_idx

def get_pc_count_for_session(mouse_num, day_num, shuffle=False, roll=False, recenter=None):
    data = load_preprocessed_data(mouse_num, day_num)
    X, X_z = data["neural"], data["z_score"]
    if shuffle:
        X_1, X_z_1 = perform_shuffle(X[0], X_z[0], roll=roll, recenter=recenter)
        X_2, X_z_2 = perform_shuffle(X[1], X_z[1], roll=roll, recenter=recenter)
        X_1[:, 0, :] = 0.
        X_z_1[:, 0, :] = 0.
        if len(X_2) > 0:
            X_2[:, 0, :] = 0.
            X_z_2[:, 0, :] = 0.
        X = [X_1, X_2]
        X_z = [X_z_1, X_z_2]
    if day_num in (1, 2, 3, 4, 6):
        pc_count = get_pc_count_from_data(X[0], X_z[0], X[1], X_z[1])
    else:
        pc_count = get_pc_count_from_data(X[0], X_z[0])
    return pc_count

def get_pc_count_for_day(day_num, shuffle=False, roll=False, recenter=None):
    return np.sum([get_pc_count_for_session(i, day_num, shuffle=shuffle, roll=roll, recenter=recenter)
                   for i in range(1, 6)])

#Helper fn.
#linregress result format: 0=slope, 1=intercept, 2=rvalue, 3=pvalue, 4=stderr
def linregress_nan(x, y):
    not_nan_mask = (~np.isnan(x)) & (~np.isnan(y))
    return linregress(x[not_nan_mask], y[not_nan_mask])

def linregress_bootstrap(x, y, num_bootstraps=10000):
    not_nan_mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[not_nan_mask], y[not_nan_mask]
    lr_bootstrap_results = np.zeros((num_bootstraps, 5))
    lr_result = linregress(x, y)
    for i in range(num_bootstraps):
        idx = np.random.choice(np.arange(len(x)), len(x), replace=True)
        lr_bootstrap_results[i] = linregress(x[idx], y[idx])
    boostrap_slopes = lr_bootstrap_results[:, 0]
    #print(np.sum(boostrap_slopes > 0) / num_bootstraps)
    return lr_result, lr_bootstrap_results

def print_linregress(result_arr):
    keys = ["slope", "intercept", "rvalue", "pvalue", "stderr"]
    to_print = ", ".join([keys[i]+": " + str(result_arr[i]) for i in range(5)])
    print(to_print)

def bin_xy(x, y, x_int, num_bins=10):
    """
    Computes the mean/sem of y in bins of x.
    
    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    x_int : 2-tuple of ints
    num_bins : int
    
    Returns
    -------
    bin_centers : np.ndarray
    y_mean : np.ndarray
    y_sem : np.ndarray
    """
    x_range = x_int[1] - x_int[0]
    x_bin_size = x_range / num_bins
    bin_centers = (np.arange(num_bins) + .5) * x_bin_size
    y_mean = np.zeros(num_bins)
    y_sem = np.zeros(num_bins)
    for i in range(num_bins):
        i1 = x_int[0] + i*x_bin_size
        i2 = x_int[0] + (i+1)*x_bin_size 
        if i < num_bins - 1:
            in_range_idx = (x >= i1) & (x < i2)
        else:
            in_range_idx = (x >= i1) & (x <= i2)
        y_in_range = y[in_range_idx]
        y_mean[i] = np.mean(y_in_range)
        y_sem[i] = sem(y_in_range)
    return bin_centers, y_mean, y_sem


def compute_licking_ratio(lick_rate):
    #To get lick_rate, call load_all_binned_behavior_data("lick_rate")
    #(5, 8, 2, 25, 40) --> (5, 8, 50, 40)
    lick_rate_flattened_trial_type = lick_rate.reshape((5, 8, 50, 40))
    X = lick_rate_flattened_trial_type #for brevity
    zero_licking_ratio_trials = np.all(X[:, :, :, -3:] == 0, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        licking_ratio = np.nanmean(X[:, :, :, -3:], axis=-1) / np.nanmean(X[:, :, :, 10:], axis=-1) 
    licking_ratio[zero_licking_ratio_trials] = 0.
    return licking_ratio

def compute_licking_com(lick_rate, idx_1=10):
    #To get lick_rate, call load_all_binned_behavior_data("lick_rate")
    #(5, 8, 2, 25, 40) --> (5, 8, 50, 40)
    lick_rate_flattened_trial_type = lick_rate.reshape((5, 8, 50, 40))
    X = lick_rate_flattened_trial_type #for brevity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.arange(idx_1, 40)[np.newaxis, np.newaxis, np.newaxis, :]
        licking_com = (np.nansum(X[:, :, :, idx_1:]*D, axis=-1)
            / np.nansum(X[:, :, :, idx_1:], axis=-1))
    return licking_com


def load_all_binned_behavior_data(vel_or_lick_rate):
    """
    Returns a big array containing the binned 'vel' or 'lick_rate' information
    for all mice.

    Returns
    -------
    data : np.ndarray, shape (5, 8, 2, 25, 40)
        Data array, where the dimensions correspond to
            0: mice
            1: days (starts at -2)
            2: trial type
            3: trial number
            4: spatial bin
    """
    data = np.ones((5, 8, 2, 25, 40)) * np.nan
    day_nums = np.arange(-2, 6)
    key = vel_or_lick_rate
    for mouse_num in range(1, 6):
        print(mouse_num)
        for day_idx in range(len(day_nums)):
            day_num = day_nums[day_idx]
            mouse_data = load_and_bin_data(mouse_num, day_num)
            trial_types = (0, 1) if day_num in (1, 2, 3, 4, 6) else (0,)
            for trial_type in trial_types:
                trials = mouse_data[key][trial_type]
                good_trial_idx = np.all(~np.isnan(trials), axis=1)
                data[mouse_num-1, day_idx, trial_type] = \
                    trials[good_trial_idx][5:30]
    return data

def load_all_binned_pc_activity(z_score=False):
    """
    Returns a list, each element of which contains the activity of all PCs on a 
    single day.

    Returns
    -------
    X_pc_all_days : list
        List of length 6. Each element is a np.ndarray with shape
        (N_pc, N_trials, N_bins) = (N_pc, 25, 40).
    """
    X_pc_all_days = []
    for day_num in range(7):
        print("day ", day_num)
        X_pc_all_mice = None
        for mouse_num in range(1, 6):
            X_pc_both = load_binned_pc_activity(mouse_num, day_num, z_score)
            #(25, 40, N_pc) --> (N_pc, 25, 40)
            X_pc_both = [np.rollaxis(X, -1) for X in X_pc_both]
            X_pc = np.concatenate(X_pc_both, axis=0)
            if X_pc_all_mice is None:
                X_pc_all_mice = X_pc
            else:
                X_pc_all_mice = np.concatenate((X_pc_all_mice, X_pc), axis=0)
        X_pc_all_days.append(X_pc_all_mice)
    return X_pc_all_days

def load_single_trial_pc_activity(mouse_num, day_num, trial_type):
    data = load_single_trial_data(mouse_num, day_num, stop_pad=0)
    X_temporal = data["neural"][trial_type][5:30]
    pos_temporal = data["x"][trial_type][5:30]
    pc_idx = get_pc_idx_for_session(mouse_num, day_num, trial_type)
    X_pc_temporal = [x[:, pc_idx] for x in X_temporal]
    return X_pc_temporal, pos_temporal


def load_binned_pc_activity(mouse_num, day_num, z_score=False):
    """
    Loads the neural activity for just the PCs.

    Parameters
    ----------
    mouse_num : int, must be one of (1, 2, 3, 4, 5)
    day_num : int, must be one of (0, 1, 2, 3, 4, 5)

    Returns
    -------
    X_pc_both : list
        List of length 2 (if day_num is in (1, 2, 3, 4, 6)) or length 1 otherwise.
        Each list element is a np.ndarray with shape
        (N_trials, N_bins, N_pc) = (25, 40, N_pc).
    """
    data = load_preprocessed_data(mouse_num, day_num)
    trial_types = (0, 1) if day_num in (1, 2, 3, 4, 6) else (0,)
    X_pc_both = []
    for trial_type in trial_types:
        X = data["neural"][trial_type]
        X_z = data["z_score"][trial_type]
        pc_idx = get_pc_idx(X, X_z)
        X_pc = X[:, :, pc_idx] if not z_score else X_z[:, :, pc_idx]
        X_pc_both.append(X_pc)
    return X_pc_both

def get_pc_density(mouse_num, day_num):
    data = load_preprocessed_data(mouse_num, day_num)
    trial_types = (0, 1) if day_num in (1, 2, 3, 4, 6) else (0,)
    sum_of_densities = 0
    for trial_type in trial_types:
        X = data["neural"][trial_type] #(25, 40, N_cells)
        N_tot = X.shape[2]
        X_z = data["z_score"][trial_type]
        pc_idx = get_pc_idx(X, X_z)
        X_pc = X[:, :, pc_idx]
        X_pc_avg = X_pc.mean(axis=0)
        centers = np.argmax(X_pc_avg[:], axis=0) #+ 1
        density = np.array([np.sum(centers == i) for i in range(40)]) / N_tot
        sum_of_densities += density
    mean_density = sum_of_densities / len(trial_types)
    return mean_density

def get_pc_density_for_day(day_num):
    return np.array([get_pc_density(mouse_num, day_num) for mouse_num in range(1, 6)]).mean(axis=0)

def get_pc_mask_subsection(X, X_trial_avg, above_frac, subsec, probf=.25,
    peakf=3., just_check_mean=False):
    mask_1 = X_trial_avg[subsec, :].max(axis=0) > peakf*X_trial_avg.mean(axis=0)
    if just_check_mean:
        return mask_1
    mask_2 = above_frac[subsec, :].max(axis=0) > probf
    mask = mask_1 & mask_2
    return mask

def get_pc_idx(X, X_z, zf=1.):
    active_mask = np.sum(X, axis=(0, 1)) > 0
    #select place cells
    above_frac = (X_z > zf).mean(axis=0)
    X_trial_avg = X.mean(axis=0)
    r1 = range(1, 11)
    r2 = range(11, 20)
    r3 = range(20, 31)
    r4 = range(31, 40)
    mask_1 = get_pc_mask_subsection(X, X_trial_avg, above_frac, r1)
    mask_2 = get_pc_mask_subsection(X, X_trial_avg, above_frac, r2)
    mask_3 = get_pc_mask_subsection(X, X_trial_avg, above_frac, r3)
    mask_4 = get_pc_mask_subsection(X, X_trial_avg, above_frac, r4)
    mask = np.array([mask_1, mask_2, mask_3, mask_4])
    pc_mask = mask.sum(axis=0) > 0
    
    #select odor cells (to exclude)
    mask_2_just_mean = get_pc_mask_subsection(X, X_trial_avg, above_frac, r2,
        just_check_mean=True)
    mask_4_just_mean = get_pc_mask_subsection(X, X_trial_avg, above_frac, r4,
        just_check_mean=True)
    centers = np.argmax(X_trial_avg[1:], axis=0) + 1
    odor_range_mask = (((centers >= r2[0]) & (centers <= r2[-1]))
        | ((centers >= r4[0]) & (centers <= r4[-1])))
    odor_mask = mask_2_just_mean & mask_4_just_mean & odor_range_mask
    
    tot_mask = active_mask & pc_mask & np.logical_not(odor_mask)
    pc_idx = np.arange(X.shape[2])[tot_mask]
    return pc_idx

def get_attr(X, key):
    """Helper for parsing data.

    Parameters
    ----------
    X : np.ndarray
    key: string

    Returns
    -------
    X_attr : np.ndarray
        Row of input 'X' corresponding to attribute 'key'
    """
    keys = ["TTLtotalCount", "Time", "Valve", "LickCount", "RewardCount",
        "InitialDropCount", "RewardWindow", "Distance", "TotalDistance",
        "LapCount", "Environment", "Velocity"]
    X_attr = X[keys.index(key)]
    return X_attr

def load_and_bin_data(mouse_num, day_num, binarize=False):
    """Loads and bins data for pretraining or imaging days. For imaging days,
    equivalent data (up to the first bin not being zero'd out) is returned by
    calling load_preprocessed_data.

    Parameters
    ----------
    mouse_num : int, must be one of (1, 2, 3, 4, 5)
    day_num : int, must be <= 5

    Returns
    -------
    binned_results : dict
        dict whose keys are:
            ["vel", "lick_rate", "neural"] if day_num >= 0
            ["vel", "lick_rate"] otherwise.
        Each entry has the form
            binned_results[key] = [T1, T2],
        where T1 and T2 are np.ndarray's of trials for types 1 or 2:
            T1 = (N_1_trials, 40),
            T2 = (N_2_trials, 40).
    """
    #large stop_pad is to simulate stop_pad=inf
    results = load_single_trial_data(mouse_num, day_num, stop_pad=1000000, binarize=binarize)
    keys_to_bin = ["vel", "lick_rate"]
    if day_num in (0, 1, 2, 3, 4, 5):
        keys_to_bin.append("neural")
    binned_results = {key:[[], []] for key in keys_to_bin + ["occupancy_count"]}
    for trial_type in range(2):
        num_trials = len(results["x"][trial_type])
        if num_trials == 0:
            continue
        for trial_idx in range(num_trials):
            x = results["x"][trial_type][trial_idx]
            occupancy_count = np.array([np.sum((x >= i*100) & (x < (i+1)*100)) for i in range(40)])
            variables = [results[key][trial_type][trial_idx]
                for key in keys_to_bin]
            binned_variables = bin_single_trial(x, variables)
            for i in range(len(keys_to_bin)):
                binned_results[keys_to_bin[i]][trial_type]\
                    .append(binned_variables[i])
            binned_results["occupancy_count"][trial_type].append(occupancy_count)
    binned_results = {key:[np.array(val[0]), np.array(val[1])]
        for _, (key, val) in enumerate(binned_results.items())}
    return binned_results

def load_preprocessed_data(mouse_num, day_num, load_day_5_pinene=False):
    """Loads pre-binned data for imaging days.

    Parameters
    ----------
    mouse_num : int, must be one of (1, 2, 3, 4, 5)
    day_num : int, must be one of (0, 1, 2, 3, 4, 5)

    Returns
    -------
    binned_results : dict
        dict whose keys are:
            ["z_score", "vel", "lick_rate", "neural"]
        Each entry has the form
            binned_results[key] = [T1, T2],
        where T1 and T2 are np.ndarray's of trials for types 1 or 2:
            T1 = (N_1_trials, 40),
            T2 = (N_2_trials, 40).
    """
    day_str = str(day_num)
    if day_num == 0:
        day_dir = "No_odor_0/"
    elif day_num == 5:
        day_dir = "Odor_NoOdor_5/"
    elif day_num == 6:
        day_dir = "Odor_NoReward_6/"
        day_str = "NoReward"
    else:
        day_dir = "Odor_day" + day_str + "_" + day_str + "/"
    keys = ["z_score", "vel", "lick_rate", "neural"] if day_num != 6 else ["z_score", "neural"]
    results = {key:[] for key in keys}
    trial_types = (0, 1) if day_num in (1, 2, 3, 4, 6) else (0,)
    if day_num == 5 and load_day_5_pinene:
        trial_types = (0, 1)
    for trial_type in trial_types:
        odor_num = trial_type + 1
        odor_str = "a1" if odor_num == 1 else "a2"
        z_score_keyname = (odor_str + "Binz_" + str(int(mouse_num)) + "_" + day_str)
        speed_keyname = ("speed" + str(odor_num) + "Bin25_" + str(int(mouse_num)) + "_" + day_str)
        lick_keyname = ("lick" + str(odor_num) + "Bin25_" + str(int(mouse_num)) + "_" + day_str)
        neural_keyname = (odor_str + "Bin25_" + str(int(mouse_num)) + "_" + day_str)
        keynames = [z_score_keyname, speed_keyname, lick_keyname, neural_keyname] if day_num != 6 \
            else [z_score_keyname, neural_keyname]
        for i in range(len(keynames)):
            keyname, key = keynames[i], keys[i]
            X = h5py.File(IND_MICE_DIR + day_dir + keyname
                + ".mat", "r")[keyname][:]
            if key in ("vel", "lick_rate", "z_score"):
                X = np.swapaxes(X, 0, 1)
            results[key].append(X)
    if day_num in (0, 5):
        for key in keys: results[key].append(np.array([]))
    return results

def load_single_trial_data(mouse_num, day_num, stop_pad=10, subset_trials=True, binarize=False):
    """Loads single-trial data for pretraining or imaging days.

    Parameters
    ----------
    mouse_num : int, must be one of (1, 2, 3, 4, 5)
    day_num : int, must be <= 5

    Returns
    -------
    data : dict
        dict returned from 'load_single_trial_data_from_filename'
        (see comments for a description of the dict's structure)
    """
    filename = get_single_trial_filename(mouse_num, day_num,
        subset_trials=subset_trials)
    data = load_single_trial_data_from_filename(filename, day_num,
        stop_pad=stop_pad, binarize=binarize)
    return data

def get_single_trial_filename(mouse_num, day_num, subset_trials=True):
    """Returns the path to the file containing the single-trial data for pretraining or imaging days.

    Parameters
    ----------
    mouse_num : int, must be one of (1, 2, 3, 4, 5)
    day_num : int, must be <= 6
    subset_trials : whether or not to load 'WS1_subset' files

    Returns
    -------
    filename : string
        Path to the mouse/day file of interest.
    """
    if day_num >= 0:
        mouse_name = MOUSE_NAMES[mouse_num-1]
        mouse_dir = RAW_IMAGING_DATA_DIR + mouse_name + "/"
        potential_subdirs = [x[0] for x in os.walk(mouse_dir)]
        if subset_trials:
            #get subdirs with WS1.mat
            subdirs = [subdir for subdir in potential_subdirs
                if "WS1_subset.mat" in os.listdir(subdir)
                or "WS1_subset30.mat" in os.listdir(subdir)]
            #sort accoridng to dates
            subdirs = np.sort(subdirs)
            subdir = subdirs[day_num]
            if day_num == 0:
                filename = subdir + "/WS1_subset30.mat"
            else:
                filename = subdir + "/WS1_subset.mat"
        else:
            #get subdirs with combined_behavior_and_s.csv
            subdirs = [subdir for subdir in potential_subdirs
                if "combined_behavior_and_s.csv" in os.listdir(subdir)]
            #sort accoridng to dates
            subdirs = np.sort(subdirs)
            subdir = subdirs[day_num]
            filename = subdir + "/combined_behavior_and_s.csv"
    else:
        #get subdirs with WS1.mat
        mouse_str = "m" + str(mouse_num)
        mouse_dir = PRETRAINING_DATA_DIR + mouse_str
        potential_subdirs = [x[0] for x in os.walk(mouse_dir)]
        subdirs = [subdir for subdir in potential_subdirs
            if "WS1.mat" in os.listdir(subdir)]
        #chop off 'pre' to get day vals (strings like '2', '3', '2a', '2b')
        day_vals = [d.split("/")[-1][3:] for d in subdirs]
        #on days with a and b, choose 'a'
        day_vals = [day_val[0] for day_val in day_vals if day_val[-1] != "b"]
        subdirs = [dir_name for dir_name in subdirs if dir_name[-1] != "b"]
        filename = subdirs[day_vals.index(str(abs(day_num)))] + "/WS1.mat"
    return filename

def load_single_trial_data_from_filename(filename, day_num, stop_pad=10, binarize=False):
    """Parses single-trial data for pretraining or imaging days given the filename.

    Parameters
    ----------
    filename : string
    day_num  : int, must be <= 6
    stop_pad : int
    binarize : boolean

    Returns
    -------
    results : dict
        dict whose keys are
            ["t", "x", "vel", "lick_rate", "neural"] if day_num >= 0
            ["t", "x", "vel", "lick_rate"] otherwise.
        Each dict entry has the form
            results[key] = [T1, T2],
        where T1 and T2 are lists of trials for types 1 or 2:
            T1 = (N_1_trials,),
            T2 = (N_2_trials,).
    """
    #Load file.
    ext = filename.split(".")[-1]
    if ext == "mat":
        f = h5py.File(filename, "r")
        data = f["WS1"][:]
    elif ext == "csv":
        data = np.loadtxt(open(filename, "rb"), delimiter=",")
        data = data.T
    else:
        return 
    #Extract attributes.
    x = get_attr(data, "Distance") #/ 1000.
    t = get_attr(data, "Time")
    vel = get_attr(data, "Velocity")
    laps = get_attr(data, "LapCount").astype(np.int)
    num_laps = np.max(laps) + 1
    environment = get_attr(data, "Environment").astype(np.int)
    lick_count = get_attr(data, "LickCount")
    #If we're on a pretraining day, then the values in 't' won't be spaced at 50 ms.
    #Thus, we interpolate all of the data at 50 ms-spaced intervals.
    if day_num < 0:
        t_even = np.floor(np.linspace(t[0], t[-1], int(np.ceil((t[-1]-t[0])/50.))))
        x = np.interp(t_even, t, x)
        vel = np.interp(t_even, t, vel)
        laps = np.round(np.interp(t_even, t, laps))
        environment = np.round(np.interp(t_even, t, environment)).astype(np.int)
        lick_count = np.round(np.interp(t_even, t, lick_count))
        t = t_even
    #Compute lick rate
    lick_rate = np.diff(lick_count) / 50.
    lick_rate = np.concatenate(([0], lick_rate))
    if day_num >= 0:
        neural = data[12:].T
        
    t_trials = []
    x_trials = []
    vel_trials = []
    lick_rate_trials = []
    trial_type = np.zeros(num_laps)
    neural_trials = []
    for lap_idx in range(num_laps):
        in_lap_mask = laps == lap_idx
        if np.sum(in_lap_mask) == 0:
            trial_type[lap_idx] = -1
            continue
        start_idx = np.argmax(in_lap_mask)
        x_in_lap = x[in_lap_mask]
        stop_idx = start_idx + np.argmax(x_in_lap > 4000)
        stop_idx += stop_pad #some padding...
        in_lap_mask[stop_idx:] = False
        if np.sum(in_lap_mask) == 0:
            trial_type[lap_idx] = -1
            continue
        t_trials.append(t[in_lap_mask])
        x_trials.append(x[in_lap_mask])
        vel_trials.append(vel[in_lap_mask])
        lick_rate_trials.append(lick_rate[in_lap_mask])
        trial_type[lap_idx] = environment[in_lap_mask][0]
        if day_num >= 0:
            if binarize:
                neural_trials.append((neural[in_lap_mask] > 1e-8).astype(np.float))
            else:
                neural_trials.append(neural[in_lap_mask])
    r = np.arange(num_laps)
    trials_1 = r[trial_type == 1]
    trials_2 = r[trial_type == 2]
    results = {}
    results["t"] = [[t_trials[i] for i in trials_1],
        [t_trials[i] for i in trials_2]]
    results["x"] = [[x_trials[i] for i in trials_1],
        [x_trials[i] for i in trials_2]]
    results["vel"] = [[vel_trials[i] for i in trials_1],
        [vel_trials[i] for i in trials_2]]
    results["lick_rate"] = [[lick_rate_trials[i] for i in trials_1],
        [lick_rate_trials[i] for i in trials_2]]
    if day_num >= 0:
        results["neural"] = [[neural_trials[i] for i in trials_1],
            [neural_trials[i] for i in trials_2]]
    return results

def bin_single_trial(x, variables):
    """Spatially bin a single trial's worth of variables.
    Uses 100 mm bins. Bins over 0--4000 mm.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Position time series measured in mm.
    variables  : list of np.ndarray's, shape (T,) or (T, N_i)
        List of scalar or vector time series.

    Returns
    -------
    binned_variables : list of np.ndarray's, shape (40,) or (40, N_i)
    """
    bin_size = 100 #spatial bins are 100 mm wide
    num_variables = len(variables)
    binned_variables = []
    for var_idx in range(num_variables):
        var = variables[var_idx]
        if len(var.shape) == 1:
            binned_variables.append(np.zeros(40))
        else:
            binned_variables.append(np.zeros((40, var.shape[1])))
    for bin_idx in range(40):
        #compute in-bin indices
        x_min = bin_size * bin_idx
        x_max = bin_size * (bin_idx + 1)
        in_bin_mask = (x >= x_min) & (x < x_max)
        nonempty = np.sum(in_bin_mask) > 0
        for var_idx in range(num_variables):
            if nonempty:
                binned_variables[var_idx][bin_idx] = np.mean(variables[var_idx][in_bin_mask], axis=0)
            else:
                binned_variables[var_idx][bin_idx] = np.nan
    return binned_variables


def form_lag_matrix(X, T, stride=1):
    """Form a time-delay embedding of X.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Input data.
    T : int
        Number of lags.
    stide : int
        Number of skips between each lagged entry.

    Returns
    -------
    X_with_lags : np.ndarray
        Lagged version of X.
    """
    N = X.shape[1]
    num_lagged_samples = floor((len(X) - T)/stride) + 1 
    X_with_lags = np.zeros((num_lagged_samples, T*N))
    for i in range(num_lagged_samples):
        X_with_lags[i, :] = X[i*stride : i*stride + T, :].flatten()
    return X_with_lags





