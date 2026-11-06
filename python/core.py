# importing modules
# %matplotlib widget # commented out as it caused error for .py script and it functions ok without this line for .py.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import subplot
from matplotlib.colors import LogNorm
from matplotlib.widgets import SpanSelector
from tqdm.auto import tqdm
from scipy.optimize import leastsq
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

# import original modules
import util

# Set environment variables for OpenCV
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"

# Setting the graph design
# plt.style.use('science')

def core(qs, dts, ISF, params, matrixfit, iqmin, iqmax, tmax, kB, T, mu, additional_plots, ballistic, no_p_size, param_col, label=None):

  # checking what is inside params
  # for i in range(0, params.shape[-1]):
  #   plt.scatter(qs,params[:,i], label ='params{}'.format(i))

  # plt.xlabel('q')
  # plt.ylabel('params')
  # plt.xscale('log')
  # plt.yscale('log')
  # plt.legend()
  # plt.title('Inside params')
  # plt.show()
    

# show A and B
  if additional_plots:
    # show A and B
    util.plot_AB(qs, params, iqmin, iqmax, label)

    # result of the fit function
    util.plot_result_fit(dts, qs, matrixfit, ISF, iqmin, iqmax, tmax, label)

    # ISF for various q for function of tau and q**2r
    util.plot_isf_4_q(dts, qs, params, ISF, iqmin, iqmax, tmax, no_p_size, param_col, label)

  X= 'None'
  alpha = 'None'
  D ='None'
  predicted_a='None'
  Xr = 'None'
  alphar = 'None'
  v ='None'

  if not ballistic:
    #free exponent fitting
    X, alpha = util.fr_exp_fit(qs, params, iqmin, iqmax, param_col, label, neglect_negative=True)

    print('\n')
    print('Free exponent fitting results:')
    print(f"alpha = {alpha:.02f}")
    print(f"X = {X:.02f}")
    print('\n')

    # Brownian motion
    print('param_col:', param_col)
    D = util.brown_d(qs, params, iqmin, iqmax, param_col, label, neglect_negative=True)
    print('Brownian results:')
    print(f"D = {D:.02f} µm²/s")


    # estimate the size of the particle
    predicted_a = kB*T/(3*np.pi*mu*D) *1e12*1e6
    print(f"Precicted Diameter = {predicted_a:.02f} µm")
    print('\n')

    Xr, alphar = util.fit_ransac(qs, params, iqmin, iqmax, param_col, label, neglect_negative=True)
    print('X ransac:', Xr)
    print('alpha ransac', alphar)

    predicted_a_ransac_free = kB*T/(3*np.pi*mu*Xr) *1e12*1e6
    print(f"Precicted Diameter = {predicted_a_ransac_free:.02f} µm")
    print('\n')


    X_incn, alpha_incn = util.fr_exp_fit(qs, params, iqmin, iqmax, param_col, label, neglect_negative=False)
    D_incn = util.brown_d(qs, params, iqmin, iqmax, param_col, label, neglect_negative=False)
    # Xr_incn, alphar_incn = util.fit_ransac(qs, params, iqmin, iqmax, param_col, label, neglect_negative=False)

    # summary plot
    plt.figure(figsize=(8,6))
    plt.plot(qs, params[:,param_col], 'o')
    plt.plot(qs, params[:,param_col], '-k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\,(\mu m^{-1})$')
    plt.ylabel(r'Characteristic time $\tau_c (s)$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))

    plt.plot([qs[iqmin], qs[iqmax]], 1/(X*np.array([qs[iqmin], qs[iqmax]])**alpha), '-r', label='Free exponent fitting with ordinary leastsq excluding negative tauc')
    plt.plot([qs[iqmin], qs[iqmax]], 1/(D*np.array([qs[iqmin], qs[iqmax]])**2), '-g', label='Brown fitting with ordinary leastsq excluding negative tauc')
    plt.plot([qs[iqmin], qs[iqmax]], 1/(Xr*np.array([qs[iqmin], qs[iqmax]])**alphar), '-m', label='Free exponent fitting with RANSAC excluding negative tauc')

    plt.plot([qs[iqmin], qs[iqmax]], 1/(X_incn*np.array([qs[iqmin], qs[iqmax]])**alpha_incn), '--r', label='Free exponent fitting with ordinary leastsq including negative tauc')
    plt.plot([qs[iqmin], qs[iqmax]], 1/(D_incn*np.array([qs[iqmin], qs[iqmax]])**2), '--g', label='Brown fitting with ordinary leastsq including negative tauc')
    # plt.plot([qs[iqmin], qs[iqmax]], 1/(Xr_incn*np.array([qs[iqmin], qs[iqmax]])**alphar_incn), '--m', label='Free exponent fitting with RANSAC including negative tauc')

    
    plt.title('Summary plot: tauc vs q')
    plt.legend()

    plt.show()



  # ballistic motion 

  if ballistic:
    v = util.ballistic(qs, params, iqmin, iqmax, param_col, label)
    print('Ballistic results:')
    print(f"v = {v:.02f} µm²/s")
    print('\n')
    print('\n')

  return X, alpha, D, predicted_a, Xr, alphar, predicted_a_ransac_free, v