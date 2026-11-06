# reference https://teclado.com/30-days-of-python/python-30-day-21-multiple-files/
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import subplot
from matplotlib.colors import LogNorm
from matplotlib.widgets import SpanSelector
from tqdm.auto import tqdm
from scipy.optimize import leastsq
# least_squares
from scipy.optimize import least_squares
# Lacked in the original code 
from matplotlib import cm
# Using pathlib
from pathlib import Path
# Using pandas
import pandas as pd
# searching for characters
import re
# scienceplots
import scienceplots
# sklearn for ransac
import sklearn

# import original modules
import util

####### classes and functions defined#######
class ImageStack(object):
    def __init__(self, filename, channel = None):
        # Initialize the ImageStack object
        self.filename = filename
        self.video = cv2.VideoCapture(filename)  # Open the video file
        property_id = int(cv2.CAP_PROP_FRAME_COUNT)  
        length = int(self.video.get(property_id))  # Get the number of frames in the video
        self.frame_count = length  # Store the number of frames
        self.fps = self.video.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
        self.channel = channel  # Store the specified channel
        self.shape = self[0].shape

    def __len__(self):
        return self.frame_count  # Return the number of frames in the video
            
    def __getitem__(self, t):
        if t<0: t= len(self)+t-1  # Handle negative indices
        assert t < self.frame_count  # Check if the index is within the range
        self.video.set(cv2.CAP_PROP_POS_FRAMES, t-1)  # Set the position of the video to the desired frame
        success, image = self.video.read()  # Read the frame

        if self.channel is not None:
            return image[...,channel]  # Return the specified channel of the frame
        if image is not None:
            return image.mean(axis=2).astype(int)  # Return the grayscale version of the frame
        self.shape = self[0].shape  # Get the shape of the first frame

# radial averager
class RadialAverager(object):
    """Radial average of a 2D array centred on (0,0), like the result of fft2d."""
    def __init__(self, shape):
        """A RadialAverager instance can process only arrays of a given shape, fixed at instanciation."""
        assert len(shape)==2
        #matrix of distances
        self.dists = np.sqrt(np.fft.fftfreq(shape[0])[:,None]**2 +  np.fft.fftfreq(shape[1])[None,:]**2)
        #dump the cross
        self.dists[0] = 0
        self.dists[:,0] = 0
        #discretize distances into bins
        self.bins = np.arange(max(shape)/2+1)/float(max(shape))
        #number of pixels at each distance
        self.hd = np.histogram(self.dists, self.bins)[0]
    
    def __call__(self, im):
        """Perform and return the radial average of the specrum 'im'"""
        assert im.shape == self.dists.shape
        hw = np.histogram(self.dists, self.bins, weights=im)[0]
        return hw/self.hd


### spectrum Diff
def spectrumDiff(im0, im1):
    """Compute the squared modulus of the 2D Fourier Transform of the difference between im0 and im1"""
    return np.abs(np.fft.fft2(im1-im0.astype(float)))**2


## timeAveraged
def timeAveraged(stack, dt, maxNCouples=20):
    """Does at most maxNCouples spectreDiff on regularly spaced couples of images. 
    Separation within couple is dt."""
    #Spread initial times over the available range
    increment = max([(len(stack)-dt)/maxNCouples, 1])
    initialTimes = np.arange(0, len(stack)-dt, increment)
    #perform the time average
    avgFFT = np.zeros(stack.shape)
    failed = 0
    for t in initialTimes:
        im0 = stack[t]
        im1 = stack[t+dt]
        if im0 is None or im1 is None:
            failed +=1
            continue
        avgFFT += spectrumDiff(im0, im1)
    return avgFFT / (len(initialTimes)-failed)

# log spaced

def logSpaced(L, pointsPerDecade=15):
    """Generate an array of log spaced integers smaller than L"""
    nbdecades = np.log10(L)
    return np.unique(np.logspace(
        start=0, stop=nbdecades, 
        num=int(nbdecades * pointsPerDecade), 
        base=10, endpoint=False
        ).astype(int))

# calculation of isf
def calculate_isf(stack, idts, maxNCouples=1000):
    """Perform time averaged and radial averaged DDM for given time intervals.
    Returns isf"""
    ra = RadialAverager(stack.shape)
    isf = np.zeros((len(idts), len(ra.hd)))
    for i, idt in tqdm(enumerate(idts), total=len(idts)):
        isf[i] = ra(timeAveraged(stack, idt, maxNCouples))
    return isf




# display isf
def plot_isf(dts, qs, ISF):
    plt.figure(figsize=(5,5))
    ISF_transposed = np.transpose(ISF)
    plt.imshow(ISF_transposed, cmap='viridis', aspect='auto', extent=[dts[0], dts[-1], qs[-1], qs[0]], norm=LogNorm())
    plt.colorbar(label='I(q,$\\tau$),[a.u.]')
    plt.title('Image Structure Function I(q,$\\tau$)')
    plt.xlabel('Lag time ($\\tau$) [s]')
    plt.ylabel('Spatial Frequency (q) [$\\mu m ^{-1}$]')
    plt.show()


# ISF fitting
def isf_fitting(LogISF, dts, ISF, tmax):
    params = np.zeros((ISF.shape[-1], 3))
    matrixfit = np.zeros(ISF[:tmax].T.shape)
    for iq, ddm in enumerate(ISF[:tmax].T):
        params[iq] = leastsq(
            # function to minimize
            lambda p, dts, logd: LogISF(p, dts) - logd,
            # initial parameters
            [np.ptp(ISF), ddm.min(), 1],
            # data on which to perform minimization
            args=(dts[:tmax], np.log(ddm))
        )[0] # https://www.tutorialspoint.com/scipy/scipy_optimize_leastsq_function.htm
        # print("What is in leastsq[1]?")
        # print(leastsq(
        #     # function to minimize
        #     lambda p, dts, logd: LogISF(p, dts) - logd,
        #     # initial parameters
        #     [np.ptp(ISF), ddm.min(), 1],
        #     # data on which to perform minimization
        #     args=(dts[:tmax], np.log(ddm))
        # )[1])
        # >> It is returning ier. if it is 1, 2, 3, 4, the solution was found.

        # Ensure non-negative values
        matrixfit[iq] = np.exp(LogISF(params[iq], dts[:tmax]))

    return params, matrixfit


def isf_fitting2(LogISF, dts, ISF, tmax):
    dts = np.longdouble(dts)
    ISF = np.longdouble(ISF)
    params = np.zeros((ISF.shape[-1], 5)) # 5...the number of parameters to fit. For the second particle, it is 5
    matrixfit = np.zeros(ISF[:tmax].T.shape)
    for iq, ddm in enumerate(ISF[:tmax].T):
        params[iq] = leastsq(
            # function to minimize
            lambda p, dts, logd: LogISF(p, dts) - logd,
            # initial parameters
            [np.ptp(ISF), ddm.min(), 1, 1, 0.5],
            # data on which to perform minimization
            args=(dts[:tmax], np.log(ddm))
        )[0]
        # params[iq] = least_squares(
        #             # function to minimize
        #             lambda p, dts, logd: LogISF(p, dts) - logd,
        #             # initial parameters
        #             [np.ptp(ISF), ddm.min(), 1, 1, 0.5],
        #             bounds = ([-np.inf, -np.inf,0, 0, 0], [np.inf, np.inf,np.inf, np.inf, 1]),
        #             # data on which to perform minimization
        #             args=(dts[:tmax], np.log(ddm)),
        #             max_nfev=1e10
        #         )[0]
        # Ensure non-negative values

        matrixfit[iq] = np.exp(LogISF(params[iq], dts[:tmax]))



    return params, matrixfit


# show A and B
def plot_AB(qs, params, iqmin, iqmax, label=None):
    plt.figure(figsize=(8,6))
    plt.title('Fit parameters A(q) and B(q) label: {}'.format(label)+'(If mixed, only axvspan should be different)')
    plt.plot(qs, params[:,0], 'o', label="A(q)") #p[0]...A
    plt.plot(qs, params[:,1], 'o', label="B(q)") #p[1]...B
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\ [\mu m^{-1}]$')
    plt.ylabel(r'$A(q),\, B(q)$')

    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))
    plt.legend()
    plt.ylim((1e3,1e11))
    plt.show()

# result of the fit function
def plot_result_fit(dts, qs, matrixfit, ISF, iqmin, iqmax, tmax, label=None):
    plt.figure(figsize=(14,8))
    ax1 = subplot(1,2,1)
    for i in range(0,len(dts[:tmax]),5):
        plt.plot(qs, ISF[i]/512**2, 'o', color=cm.autumn(i/float(len(dts[:tmax]))),label = f'$\\tau = {round(dts[i], 2)}s$')
        plt.plot(qs, matrixfit[:,i]/512**2, '-k')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$f(q,\tau)$') # Shouldn't it be ISF instead of f?
    plt.xlabel(r'$q\ [\mu m^{-1}]$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))
    plt.legend()
    plt.title("ISF against q Label {}".format(label)+'(If mixed, only axvspan should be different)')

    ax2 = subplot(1,2,2, sharey=ax1)
    for i,iq in enumerate(np.logspace(np.log10(iqmin),np.log10(iqmax),10, base=10).astype(int)):
        plt.plot(dts[:tmax], ISF[:tmax,iq]/512**2, 'o', color=cm.autumn(i/10.),label=f'$q = {round(qs[iq], 2)} \\mu m^{{-1}}$')
        plt.plot(dts[:tmax], matrixfit[iq]/512**2, '-k')
    plt.xscale('log')
    plt.xlabel(r'$\tau\ [s]$')
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.legend()
    plt.ylim(10, 1e5)
    plt.title("ISF against tau Label {}".format(label)+'(If mixed, only axvspan should be different)')
    plt.show()

def plot_isf_4_q(dts, qs, params, ISF, iqmin, iqmax, tmax, no_p_size, param_col=2, label=None):
    # plotting corresponding to f?
    # The graph should be the same for the first and second particle
    plt.figure(figsize=(14,8))
    ax1 = plt.subplot(1,2,1)
    for i,iq in enumerate(np.logspace(np.log10(iqmin),np.log10(iqmax),10, base=10).astype(int)):
        plt.plot(dts[:tmax], 1- (ISF[:tmax,iq]-params[iq,1])/params[iq,0], 'o', color=cm.autumn(i/10.), label=f'$q = {round(qs[iq], 2)}\\mu m^{{-1}}$') #p[0]...A #p[1]...B

        if no_p_size==1:
            plt.plot(dts[:tmax], np.exp(-dts[:tmax]/params[iq,param_col]), '-k')

        if no_p_size==2:
            # I wanted it to depend on the inputs like param_col, but I am putting like this for now.
            f = params[iq, 4]*np.exp(-dts[:tmax]/params[iq,2])+(1-params[iq, 4])*np.exp(-dts[:tmax]/params[iq,3])
            plt.plot(dts[:tmax], f, '-k') 
    plt.xscale('log')
    plt.ylabel(r'$ISF(q,\tau)$') # Shouldn't it be f instead of ISF? ISF includes A and B.
    plt.xlabel(r'$\tau\ [s]$')
    plt.legend()
    plt.title('f against tau (If mixed, this should be the same for different particles) ')

    ax2 = subplot(1,2,2, sharey=ax1)
    for i,iq in enumerate(np.logspace(np.log10(iqmin),np.log10(iqmax),10, base=10).astype(int)):
        ax2.plot(qs[iq]**2*dts[:tmax], 1- (ISF[:tmax,iq]-params[iq,1])/params[iq,0], 'o', color=cm.autumn(i/10.),label=f'$q = {round(qs[iq], 2)}\\mu m^{{-1}}$')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$q^2\tau\ [\mu m^2s]$')
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_ylim(-0.1,1.1)
    ax2.legend()
    plt.title('f against q**2tau')

    plt.show()


# free exponent fitting
def fr_exp_fit(qs, params, iqmin, iqmax, param_col=2, label=None, neglect_negative=False):
    plt.figure(figsize=(8,6))
    plt.plot(qs, params[:,param_col], 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\,(\mu m^{-1})$')
    plt.ylabel(r'Characteristic time $\tau_c (s)$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))


    # https://note.nkmk.me/python-numpy-where/ https://www.headboost.jp/numpy-delete/
    i_less_than_zeros = np.where(params[iqmin:iqmax, param_col]<=0)[0]
    qs_p_positive = np.delete(qs[iqmin:iqmax], i_less_than_zeros)
    params_positive = np.delete(params[iqmin:iqmax, param_col], i_less_than_zeros)

    if not neglect_negative:
        qs_used = qs[iqmin:iqmax]
        params_used = params[iqmin:iqmax,param_col]

    if neglect_negative:
        qs_used = qs_p_positive
        params_used = params_positive


    X, alpha = leastsq(
        lambda p, q, td: p[0] - p[1]*np.log(np.abs(q)) - np.log(np.abs(td)),
        [1,2],
        args=(qs_used, params_used)
        )[0]
    X= np.exp(-X)
    plt.plot([qs[iqmin], qs[iqmax]], 1/(X*np.array([qs[iqmin], qs[iqmax]])**alpha), '-r')
    plt.title('Free exponent fitting Label {} Neglect negative: {}'.format(label, neglect_negative))

    plt.show()

    return X, alpha

# Brownian motion fitting
def brown_d(qs, params, iqmin, iqmax, param_col=2, label=None,neglect_negative=False):
    plt.figure(figsize=(8,6))
    plt.plot(qs, params[:,param_col], 'o')
    # plt.plot(qs, params[:,param_col], '-k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\,(\mu m^{-1})$')
    plt.ylabel(r'Characteristic time $\tau_c (s)$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))

    i_less_than_zeros = np.where(params[iqmin:iqmax, param_col]<=0)[0]
    qs_p_positive = np.delete(qs[iqmin:iqmax], i_less_than_zeros)
    params_positive = np.delete(params[iqmin:iqmax, param_col], i_less_than_zeros)

    if not neglect_negative:
        qs_used = qs[iqmin:iqmax]
        params_used = params[iqmin:iqmax,param_col]

    if neglect_negative:
        qs_used = qs_p_positive
        params_used = params_positive


    D = np.exp(-leastsq(
        lambda p, q, td: p[0] - 2*np.log(q) - np.log(td),
        [-1.],
        args=(qs_used, params_used)
        )[0][0])

    # #checking 
    # print('checkiing the fit of util.brown_d. If this is not 1 2 3 4, the fit did not succeed')
    # print(leastsq(
    #     lambda p, q, td: p[0] - 2*np.log(q) - np.log(td),
    #     [-1.],
    #     args=(qs[iqmin:iqmax], params[iqmin:iqmax,param_col])
    #     )[1])
    
    plt.plot([qs[iqmin], qs[iqmax]], 1/(D*np.array([qs[iqmin], qs[iqmax]])**2), '-r')
    plt.title("Slope forced to -2(Brownian) Label {} neglect negative: {}".format(label, neglect_negative))
    # plt.ylim(np.min(params[:,param_col])*2, np.max(params[:,param_col])*2) #https://numpy.org/doc/2.2/reference/generated/numpy.min.html # Some values seem to be diverging for mixed particles
    # plt.ylim(1e-50, 1e4)

    plt.show()

    return D

# RANSAC
def fit_ransac(qs, params, iqmin, iqmax, param_col=2, label=None, neglect_negative=False): #https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html #https://qiita.com/smurakami/items/14202a83bd13e55d4c09
    plt.figure(figsize=(8,6))
    plt.plot(qs, params[:,param_col], 'o')
    plt.plot(qs, params[:,param_col], '-k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\,(\mu m^{-1})$')
    plt.ylabel(r'Characteristic time $\tau_c (s)$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))

    # Finding where the negative values are and neglecting them
    i_less_than_zeros = np.where(params[iqmin:iqmax, param_col]<=0)[0]
    qs_p_positive = np.delete(qs[iqmin:iqmax], i_less_than_zeros)
    params_positive = np.delete(params[iqmin:iqmax, param_col], i_less_than_zeros)

    if not neglect_negative:
        qs_used = qs[iqmin:iqmax]
        params_used = params[iqmin:iqmax,param_col]

    if neglect_negative:
        qs_used = qs_p_positive
        params_used = params_positive

    ransac = sklearn.linear_model.RANSACRegressor()
    ransac.fit(np.log(qs_used).reshape(-1,1), np.log(params_used))
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask) # https://numpy.org/doc/2.5/reference/generated/numpy.logical_not.html
    alpha = -ransac.estimator_.coef_
    X = np.exp(-ransac.estimator_.intercept_)
    

    plt.plot([qs[iqmin], qs[iqmax]], 1/(X*np.array([qs[iqmin], qs[iqmax]])**alpha), '-r')
    ext_qs = qs[iqmin:iqmax]
    ext_params = params[iqmin:iqmax, param_col]

    # plt.scatter(ext_qs.reshape(-1,1)[inlier_mask], ext_params[inlier_mask], '.', label='Inlier') # https://qiita.com/junjis0203/items/4136f9ae4f07c452ceb6
    # plt.scatter(ext_qs.reshape(-1,1)[outlier_mask], ext_params[outlier_mask], '.', label='Outlier') # https://qiita.com/junjis0203/items/4136f9ae4f07c452ceb6

    plt.title("RANSAC Label {}".format(label))
    plt.legend()

    plt.show()

    return X, alpha

# # RANSAC
# def brown_ransac(qs, params, iqmin, iqmax, param_col=2, label=None): #https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html #https://qiita.com/smurakami/items/14202a83bd13e55d4c09
#     plt.figure(figsize=(8,6))
#     plt.plot(qs, params[:,param_col], 'o')
#     plt.plot(qs, params[:,param_col], '-k')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel(r'$q\,(\mu m^{-1})$')
#     plt.ylabel(r'Characteristic time $\tau_c (s)$')
#     plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))

#     # How can I force the slope to 2?
#     ransac = sklearn.linear_model.RANSACRegressor()
#     ransac.fit(qs[iqmin:iqmax], params[iqmin:iqmax, param_col])
#     inlier_mask = ransac.inlier_mask_
#     outlier_mask = np.logical_not(inlier_mask) # https://numpy.org/doc/2.5/reference/generated/numpy.logical_not.html
#     X = np.exp(-ransac.estimator.intercept_)
#     alpha = -ransac.estimator.coef_
#     plt.plot([qs[iqmin], qs[iqmax]], 1/(X*np.array([qs[iqmin], qs[iqmax]])**alpha), '-r')

#     plt.title("RANSAC Label {}".format(label))

#     plt.show()












# ballistic motion
def ballistic(qs, params, iqmin, iqmax, param_col=2, label=None):
    plt.figure(figsize=(8,6))
    plt.plot(qs, params[:,param_col], 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\,(\mu m^{-1})$')
    plt.ylabel(r'Characteristic time $\tau_c (s)$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))

    v = np.exp(-leastsq(
        lambda p, q, td: p - np.log(np.abs(q)) - np.log(np.abs(td)),
        [-1],
        args=(qs[iqmin:iqmax], params[iqmin:iqmax,param_col])
        )[0][0])
    plt.plot([qs[iqmin], qs[iqmax]], 1/(v*np.array([qs[iqmin], qs[iqmax]])), '-r')
    plt.title('Ballistic fitting Label {}'.format(label))
    plt.show()

    return v


def calc_d_theory(kB,T,mu,a):
# D theory
    D_theory = kB*T/(3*np.pi*mu*a) * 1e12
    # print('raw D theory', D_theory)
    # print(f"D theory = {D_theory:.02f} µm²/s")
    return D_theory



# Code from others below

#####################################
# Author:   Guru Subramani
# Date:     April 2nd 2018
# MIT License
#####################################
from scipy.optimize import minimize


#TODO Add pythonic documentation elements

import numpy as np

def robust_lsq_ransac(model_error_func, model_fit_func, X,
               iterations=1000, fit_samples=2, fit_with_best_n=None, priors=None,norm_func = np.arctan):

    if priors == None:
        probabilities = np.ones(len(X))
    else:
        probabilities = priors

    probabilities /= np.sum(probabilities)
    indices = np.array(range(len(X)))
    current_prob = probabilities.copy()
    for iter in range(iterations):
        sampled_indices = np.random.choice(range(len(indices)),
                                           p=probabilities, size=fit_samples, replace=False)

        X_subset = X[sampled_indices]
        params = model_fit_func(X_subset)
        errors = model_error_func(X, params)

        current_prob[:] = 1 / norm_func(1 + errors[:])
        probabilities *= current_prob
        probabilities /= np.sum(probabilities)

    if fit_with_best_n == None:
        return probabilities
    else:
        robust_X = X[np.argsort(probabilities)[-fit_with_best_n:]]
        robust_params = model_fit_func(robust_X)
        return probabilities, robust_params, model_error_func(robust_X, robust_params)


def robust_lsq_m_estimates(model_error_func, model_fit_func, X,
                           iterations=1000, priors=None,norm_func = lambda x: 1/(1 + x**0.1)):
    if priors == None:
        probabilities = np.ones(len(X))
    else:
        probabilities = priors

    probabilities /= np.sum(probabilities)
    best_errors = 1e100
    best_param = None
    current_prob = probabilities.copy()
    for iter in range(iterations):
        params = model_fit_func(X, probabilities)
        errors = model_error_func(X, params, probabilities)
        if np.sum(errors) < best_errors:
            best_param = params
            best_errors = np.sum(errors)

        current_prob[:] = norm_func(errors[:])
        probabilities *= current_prob
        probabilities /= np.sum(probabilities)
    params = model_fit_func(X, probabilities)
    errors = model_error_func(X, params, probabilities)
    if np.sum(errors) < best_errors:
        best_param = params
    return probabilities, best_param, errors


def linear_model(x,y,a,b,weights):
    return (a*x - y + b)**2*weights

def fit_model(model,init_cond,x_vec,y_vec,weights):
    J = lambda param: np.sum(model(x_vec,y_vec,param[0],param[1],weights))
    sol = minimize(J,init_cond)
    return sol


# todo: