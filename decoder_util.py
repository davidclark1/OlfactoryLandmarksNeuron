"""
David Clark
December 2019
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from scipy.special import xlogy
from scipy.ndimage import gaussian_filter1d
import data_util
import matplotlib.pyplot as plt

def make_quad_features(x):
    """
    Creates quadratic features.

    Parameters
    ----------
    x : np.ndarray, shape (N,)

    Returns
    -------
    x_quad : np.ndarray, shape (N, 3)
    """
    x_quad = np.zeros((len(x), 3))
    x_quad[:, 0] = 1.
    x_quad[:, 1] = x
    x_quad[:, 2] = x**2
    return x_quad

def decode_position(mouse_num, day_num, trial_type, start_pos=0.1, pad_amt=7, return_full_err=False):
    #Load data.
    data = data_util.load_single_trial_data(mouse_num, day_num, stop_pad=0)
    x_trials = data["x"][trial_type][5:30]
    vel_trials = data["vel"][trial_type][5:30]
    neural_trials = data["neural"][trial_type][5:30]

    neural_concat = np.concatenate(neural_trials, axis=0)
    good_nrns = neural_concat.mean(axis=0) > 0 #.05
    plt.hist(neural_concat.mean(axis=0), bins=300)
    plt.show()
    print("good nrns =", np.sum(good_nrns), "of", len(good_nrns))
    neural_trials = [n[:, good_nrns] for n in neural_trials]

    for i in range(25):
        #Chop off beginning until start_pos
        start_idx = np.argmax(x_trials[i] > start_pos*1e3)
        neural_trials[i] = neural_trials[i][start_idx:]
        x_trials[i] = x_trials[i][start_idx:]
        vel_trials[i] = vel_trials[i][start_idx:]

        #Form lag matrix
        neural_trials[i] = data_util.form_lag_matrix(neural_trials[i], 2*pad_amt+1, stride=1)
        x_trials[i] = x_trials[i][pad_amt:-pad_amt]
        vel_trials[i] = vel_trials[i][pad_amt:-pad_amt]

    #Compute avg. vel.
    avg_vel_trials = np.array([np.mean(vel) for vel in vel_trials])
    #Binarize neural activity.
    neural_trials_bin = [(neural > 0).astype(np.float) for neural in neural_trials]
    #Change units of position from mm to m.
    x_trials = [x * 1e-3 for x in x_trials]
    #Run LOO-cross validation.
    dec = NaiveBayesianDecoder()
    #sigma=.2/(3.9/100.), min_spike_prob=1e-4, pad_amt=7
    x_decoded_trials = dec.cross_validate(x_trials, neural_trials_bin,
        x_int=(start_pos, 4.), N_pts=78, sigma=.2/(3.9/78.), min_spike_prob=1e-4,
        leave_one_out=True, method="histogram")

    #Compute average and RMS error.
    avg_err_trials = np.array([np.mean(x_decoded_trials[i]-x_trials[i])
                               for i in range(len(x_trials))])
    rms_err_trials = np.array([np.sqrt(np.mean((x_decoded_trials[i]-x_trials[i])**2))
                               for i in range(len(x_trials))])
    #ADDED FOR NEURON REVIEW
    err_trials = [(x_decoded_trials[i]-x_trials[i])**2 for i in range(len(x_trials))]
    binned_err_trials = np.array([np.sqrt(data_util.bin_xy(x_trials[i], err_trials[i], (0., 4.), num_bins=40)[1]) for i in range(len(x_trials))])

    if return_full_err:
        return avg_vel_trials, avg_err_trials, rms_err_trials, binned_err_trials
    else:
        return avg_vel_trials, avg_err_trials, rms_err_trials

class NaiveBayesianDecoder():
    """
    Naive Bayesian deocder for mapping neural data-->postion.
    """

    def fit(self, x_trials, y_trials, x_int, N_pts, sigma, min_spike_prob,
        method="histogram", verbose=True):
        """
        Fits the model.

        Parameters
        ----------
        x_trials : list
            i-th element: np.ndarray, shape (T_i,)
        y_trials : list
            i-th element: np.ndarray, shape (T_i, N)
        x_int : 2-tuple of upper and lower position bounds in m
        N_pts : number of points over which position values are sampled
        method : string, one of ("histogram", "gqm")
        sigma : float
        verbose : boolean
        min_spike_prob : float
        
        Returns
        -------
        None
        """
        if method not in ("histogram", "gqm", "linear"):
            raise ValueError("'method' must be one of ('histogram', 'gqm')")
        x_concat = np.concatenate(x_trials, axis=0)
        y_concat = np.concatenate(y_trials, axis=0)
        T, N = y_concat.shape
        x_concat_quad = make_quad_features(x_concat)
        #Create x-values for tuning curves.
        x_range = x_int[1] - x_int[0]
        bin_size = x_range / N_pts
        pos_vals = x_int[0] + bin_size*(np.arange(N_pts)+.5)
        pos_vals_quad = make_quad_features(pos_vals)
        #Fit the model (build quadratic curves).
        tuning_curves = np.zeros((N_pts, N))
        if method == "gqm":
            num_value_errors = 0
            num_perfect_seperation_errors = 0
            for j in range(N):
                if verbose and j % 100 == 0:
                    print(j, "of", N)
                model = sm.GLM(y_concat[:, j], x_concat_quad,
                    family=sm.families.Binomial())
                try:
                    fit_model = model.fit()
                    tuning_curve = fit_model.predict(pos_vals_quad)
                except ValueError as err:
                    num_value_errors += 1
                    tuning_curve = np.zeros(N_pts)
                except PerfectSeparationError as err:
                    num_perfect_seperation_errors += 1
                    tuning_curve = np.zeros(N_pts)
                tuning_curves[:, j] = tuning_curve
            if verbose:
                print("ValueErrors:", num_value_errors, "/", N)
                print("PerfectSeparationErrors:", num_perfect_seperation_errors,
                    "/", N)
        elif method == "histogram":
            for i in range(N_pts):
                x1 = x_int[0] + bin_size*i
                x2 = x_int[0] + bin_size*(i+1)
                in_range_mask = (x_concat >= x1) & (x_concat < x2)
                if np.sum(in_range_mask) > 0:
                    tuning_curves[i] = y_concat[in_range_mask].mean(axis=0)
            tuning_curves = gaussian_filter1d(tuning_curves, axis=0,
                sigma=sigma)
            plt.imshow(tuning_curves[:, :N//15])
            plt.show()
        elif method == "linear":
            w = np.linalg.lstsq(y_concat, x_concat.reshape((len(x_concat), 1)))[0].flatten()
            self.w = w


        #IMPORTANT: Set minimum spike prob. to min_spike_prob!
        tuning_curves = np.maximum(tuning_curves, min_spike_prob)
        #plt.imshow(tuning_curves)
        #plt.show()
        #Save variables in instance.
        self.pos_vals = pos_vals
        self.tuning_curves = tuning_curves
        
    def transform(self, y_trials, max_dx=0.085):
        """
        Uses the trained model to predict position form neural data.

        Parameters
        ----------
        y_trials : list
            i-th element: np.ndarray, shape (T_i, N)
        max_dx : int
            Maximum distance over which the decoded position can jump.
            NOTE: when max_dx=0.085, x_int=(0, 4) and N_pts=100, the state
            can either jump 0, 1, or 2 bins, so the max speed is (2 bins / 50 ms)*(40 mm / bin) = 1.6 m/s

        Returns
        -------
        x_decoded_trials : list
            i-th element: np.ndarray, shape (T_i,)
        """
        if hasattr(self, "w"):
            return [np.dot(y, self.w) for y in y_trials]

        pos_vals = self.pos_vals
        #print(self.pos_vals)
        #max_dx = max_dx_bins * np.diff(pos_vals)[0] * (1+1e-2)
        tuning_curves = self.tuning_curves
        x_decoded_trials = []
        for y in y_trials:
            #Run decoder.
            T, N = y.shape
            x_decoded = np.zeros(T)
            x_decoded[0] = self.pos_vals[0]
            for t in range(1, T):
                r = y[t] #Observed spike counts.
                log_spike_probs = np.sum((xlogy(r[np.newaxis, :], tuning_curves)
                    + xlogy(1.-r[np.newaxis, :], 1.-tuning_curves)), axis=1)
                #log_pos_probs = np.zeros(len(pos_vals))
                #dx = pos_vals - x_decoded[t-1]
                #log_pos_probs[(dx > max_dx) | (dx < 0.)] = -np.inf
                #map_est_idx = np.argmax(log_pos_probs + log_spike_probs)
                map_est_idx = np.argmax(log_spike_probs)
                map_est_pos = pos_vals[map_est_idx]
                x_decoded[t] = map_est_pos
            x_decoded_trials.append(x_decoded)
        return x_decoded_trials

    def cross_validate(self, x_trials, y_trials, x_int, N_pts, sigma, min_spike_prob,
        method="histogram", verbose=True, num_folds=5, leave_one_out=False):
        """
        Uses the trained model to perform decoding using cross-validation.

        Parameters
        ----------
        For all but the following two parameters, see 'fit'.
        num_folds : int
        leave_one_out : boolean

        Returns
        -------
        x_decoded_trials : list
            i-th element: np.ndarray, shape (T_i,)
        """
        num_trials = len(x_trials)
        if leave_one_out:
            num_folds = num_trials
        fold_size = num_trials // num_folds
        all_idx = np.arange(num_trials)
        x_decoded_trials = []
        x_trials = np.array(x_trials)
        y_trials = np.array(y_trials)
        for fold_idx in range(num_folds):
            i1 = fold_size*fold_idx
            i2 = fold_size*(fold_idx+1) if fold_idx < num_folds-1 else num_trials
            test_idx = np.arange(i1, i2)
            train_idx = np.setdiff1d(all_idx, test_idx)
            x_train = x_trials[train_idx]
            y_train = y_trials[train_idx]
            x_test = x_trials[test_idx]
            y_test = y_trials[test_idx]
            self.fit(x_train, y_train, x_int=x_int, N_pts=N_pts, sigma=sigma,
                min_spike_prob=min_spike_prob, method=method, verbose=verbose)
            x_decoded = self.transform(list(y_test))
            #x_decoded[0] = gaussian_filter1d(x_decoded[0], sigma=5)
            plt.plot(x_decoded[0])
            plt.plot(x_test[0])
            plt.xlim([0, 250])
            plt.ylim([0, 4.1])
            plt.show()
            x_decoded_trials += x_decoded
        return x_decoded_trials





