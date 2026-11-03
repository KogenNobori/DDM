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
        )[0]
        # Ensure non-negative values
        matrixfit[iq] = np.exp(LogISF(params[iq], dts[:tmax]))

    return params, matrixfit


def isf_fitting2(LogISF, dts, ISF, tmax):
    params = np.zeros((ISF.shape[-1], 3))
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
        # Ensure non-negative values
        matrixfit[iq] = np.exp(LogISF(params[iq], dts[:tmax]))

    return params, matrixfit


# show A and B
def plot_AB(qs, params, iqmin, iqmax):
    plt.figure(figsize=(8,6))
    plt.title('Fit parameters A(q) and B(q)')
    plt.plot(qs, params[:,0], 'o', label="A(q)")
    plt.plot(qs, params[:,1], 'o', label="B(q)")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\ [\mu m^{-1}]$')
    plt.ylabel(r'$A(q),\, B(q)$')

    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))
    plt.legend()
    plt.ylim((1e3,1e11))
    plt.show()

# result of the fit function
def plot_result_fit(dts, qs, matrixfit, ISF, iqmin, iqmax, tmax):
    plt.figure(figsize=(14,8))
    ax1 = subplot(1,2,1)
    for i in range(0,len(dts[:tmax]),5):
        plt.plot(qs, ISF[i]/512**2, 'o', color=cm.autumn(i/float(len(dts[:tmax]))),label = f'$\\tau = {round(dts[i], 2)}s$')
        plt.plot(qs, matrixfit[:,i]/512**2, '-k')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$f(q,\tau)$')
    plt.xlabel(r'$q\ [\mu m^{-1}]$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))
    plt.legend()

    ax2 = subplot(1,2,2, sharey=ax1)
    for i,iq in enumerate(np.logspace(np.log10(iqmin),np.log10(iqmax),10, base=10).astype(int)):
        plt.plot(dts[:tmax], ISF[:tmax,iq]/512**2, 'o', color=cm.autumn(i/10.),label=f'$q = {round(qs[iq], 2)} \\mu m^{{-1}}$')
        plt.plot(dts[:tmax], matrixfit[iq]/512**2, '-k')
    plt.xscale('log')
    plt.xlabel(r'$\tau\ [s]$')
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.legend()
    plt.ylim(10, 1e5)
    plt.show()

def plot_isf_4_q(dts, qs, params, ISF, iqmin, iqmax, tmax):
    plt.figure(figsize=(14,8))
    ax1 = plt.subplot(1,2,1)
    for i,iq in enumerate(np.logspace(np.log10(iqmin),np.log10(iqmax),10, base=10).astype(int)):
        plt.plot(dts[:tmax], 1- (ISF[:tmax,iq]-params[iq,1])/params[iq,0], 'o', color=cm.autumn(i/10.), label=f'$q = {round(qs[iq], 2)}\\mu m^{{-1}}$')
        plt.plot(dts[:tmax], np.exp(-dts[:tmax]/params[iq,2]), '-k')
    plt.xscale('log')
    plt.ylabel(r'$ISF(q,\tau)$')
    plt.xlabel(r'$\tau\ [s]$')
    plt.legend()

    ax2 = subplot(1,2,2, sharey=ax1)
    for i,iq in enumerate(np.logspace(np.log10(iqmin),np.log10(iqmax),10, base=10).astype(int)):
        ax2.plot(qs[iq]**2*dts[:tmax], 1- (ISF[:tmax,iq]-params[iq,1])/params[iq,0], 'o', color=cm.autumn(i/10.),label=f'$q = {round(qs[iq], 2)}\\mu m^{{-1}}$')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$q^2\tau\ [\mu m^2s]$')
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_ylim(-0.1,1.1)
    ax2.legend()

    plt.show()


# free exponent fitting
def fr_exp_fit(qs, params, iqmin, iqmax):
    plt.figure(figsize=(8,6))
    plt.plot(qs, params[:,2], 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\,(\mu m^{-1})$')
    plt.ylabel(r'Characteristic time $\tau_c (s)$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))

    X, alpha = leastsq(
        lambda p, q, td: p[0] - p[1]*np.log(np.abs(q)) - np.log(np.abs(td)),
        [1,2],
        args=(qs[iqmin:iqmax], params[iqmin:iqmax,2])
        )[0]
    X= np.exp(-X)
    plt.plot([qs[iqmin], qs[iqmax]], 1/(X*np.array([qs[iqmin], qs[iqmax]])**alpha), '-r')
    plt.title('Free exponent fitting')

    plt.show()

    return X, alpha

# Brownian motion fitting
def brown_d(qs, params, iqmin, iqmax):
    plt.figure(figsize=(8,6))
    plt.plot(qs, params[:,2], 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\,(\mu m^{-1})$')
    plt.ylabel(r'Characteristic time $\tau_c (s)$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))
    D = np.exp(-leastsq(
        lambda p, q, td: p[0] - 2*np.log(q) - np.log(td),
        [-1.],
        args=(qs[iqmin:iqmax], params[iqmin:iqmax,2])
        )[0][0])
    plt.plot([qs[iqmin], qs[iqmax]], 1/(D*np.array([qs[iqmin], qs[iqmax]])**2), '-r')
    plt.title("Slope forced to -2")

    plt.show()

    return D

# ballistic motion
def ballistic(qs, params, iqmin, iqmax):
    plt.figure(figsize=(8,6))
    plt.plot(qs, params[:,2], 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q\,(\mu m^{-1})$')
    plt.ylabel(r'Characteristic time $\tau_c (s)$')
    plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))

    v = np.exp(-leastsq(
        lambda p, q, td: p - np.log(np.abs(q)) - np.log(np.abs(td)),
        [-1],
        args=(qs[iqmin:iqmax], params[iqmin:iqmax,2])
        )[0][0])
    plt.plot([qs[iqmin], qs[iqmax]], 1/(v*np.array([qs[iqmin], qs[iqmax]])), '-r')
    plt.show()

    return v




