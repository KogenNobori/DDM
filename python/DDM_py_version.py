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


# Set environment variables for OpenCV
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"


############## Variables ##############
video_file = "/Volumes/T7 Shield/e1_stand1/data/Concentrationdiff2mium/rt/1-5000_10s_32p945fps_rt-0000.avi" # Change this to the name of the video file you want to analyse
pixelSize = 100/844.341 #40x (in micrometre)

pointsPerDecade = 30
maxNCouples = 10 #10 for fast evaluation, 300 for accurate analysis

kB = 1.38e-23
T = 273.15+23.1 # measured room temperature
mu = 8.9e-4
a = 2e-6 #change this to the diameter of the particle in metres

###### Choose the correlation function###

LogISF = lambda p, dts: np.log(np.maximum(p[0] * (1 - np.exp(-dts / p[2])) + p[1], 1e-10))

#LogISF = lambda p, dts: np.log(p[0] * (1-(np.sin(dts/p[2])/(dts/p[2]))) + p[1]) 



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

# onselect
def onselect(xmin, xmax):
    global iqmin, iqmax
    iqmin= np.searchsorted(qs, xmin)
    iqmax = np.searchsorted(qs, xmax)
    print(f"Selected range: {qs[iqmin]:.2f} to {qs[iqmax]:.2f}")


######### Checking the data########
stack = ImageStack(video_file)


# plt.figure(figsize=(14,4))
# plt.suptitle('Checking the Image Stack')
# subplot(1,3,1).imshow(stack[0], 'gray')
# subplot(1,3,1).set_title('First Frame')
# subplot(1,3,2).imshow(stack[-1], 'gray')
# subplot(1,3,2).set_title('Last Frame')
# subplot(1,3,3).imshow(np.abs(stack[-1]-stack[0]).astype(float), 'gray')
# subplot(1,3,3).set_title('Difference')


# plt.figure(figsize=(14,4))
# plt.suptitle('Spectra for different lag times')
# max_v = np.percentile(np.fft.fftshift(spectrumDiff(stack[0], stack[3])),99)
# subplot(1,3,1).imshow(np.fft.fftshift(spectrumDiff(stack[0], stack[3])), 'hot',vmin=0, vmax=max_v)
# subplot(1,3,1).set_title(r'$\tau = 0.1s$')
# subplot(1,3,2).imshow(np.fft.fftshift(spectrumDiff(stack[0], stack[30])), 'hot',vmin=0, vmax=max_v)
# subplot(1,3,2).set_title(r'$\tau = 1s$')
# subplot(1,3,3).imshow(np.fft.fftshift(spectrumDiff(stack[0], stack[300])), 'hot',vmin=0, vmax=max_v)
# subplot(1,3,3).set_title(r'$\tau = 10s$')


# time averaged spectra
# plt.figure(figsize=(14,4))
# plt.suptitle('Time Averaged Spectra')

# timeAverage_3 = timeAveraged(stack, 3, 5)
# timeAverage_30 = timeAveraged(stack, 30, 5)
# timeAverage_300 = timeAveraged(stack, 300, 5)

# subplot(1,3,1).imshow(np.fft.fftshift(timeAverage_3), 'hot',vmin=0, vmax=max_v)
# subplot(1,3,1).set_title('Time Average (dt = 3s)')
# subplot(1,3,2).imshow(np.fft.fftshift(timeAverage_30), 'hot',vmin=0, vmax=max_v)
# subplot(1,3,2).set_title('Time Average (dt = 30s)')
# subplot(1,3,3).imshow(np.fft.fftshift(timeAverage_300), 'hot',vmin=0, vmax=max_v)
# subplot(1,3,3).set_title('Time Average (dt = 300s)')

# radial average of the spectrum
# plt.figure(figsize=(5,5))
# ra = RadialAverager(stack.shape)
# plt.plot(ra(timeAverage_30))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('q [px$^{-1}$]')
# plt.ylabel('Intensity [a.u.]')
# plt.title('Radial Average of the Time Averaged Spectrum')

##### Outputting filename ####
file_path = Path(video_file) #https://note.nkmk.me/python-pathlib-name-suffix-parent/#google_vignette #https://docs.python.org/ja/3/library/pathlib.html#pathlib.PurePath.name #https://docs.python.org/3.15/library/pathlib.html#
print('filename:', file_path.name)

####


# calculate isf

idts = logSpaced(len(stack), pointsPerDecade)
dts = idts/stack.fps
ISF = calculate_isf(stack, idts, maxNCouples)

# calculate the scale of wavevectors
qs = 2*np.pi/(2*ISF.shape[-1]*pixelSize) * np.arange(ISF.shape[-1])
tmax = -1

# display the isf
plt.figure(figsize=(5,5))
ISF_transposed = np.transpose(ISF)
plt.imshow(ISF_transposed, cmap='viridis', aspect='auto', extent=[dts[0], dts[-1], qs[-1], qs[0]], norm=LogNorm())
plt.colorbar(label='I(q,$\\tau$),[a.u.]')
plt.title('Image Structure Function I(q,$\\tau$)')
plt.xlabel('Lag time ($\\tau$) [s]')
plt.ylabel('Spatial Frequency (q) [$\\mu m ^{-1}$]')
plt.show()


# perform the ISF fitting

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


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(qs, params[:, 2], 'o')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$q\ [\mu m^{-1}$]')
ax.set_ylabel(r'Characteristic time $\tau_c\ [s]$')
ax.set_title('Click and drag to select a valid range of q values')

span = SpanSelector(ax, onselect, 'horizontal', useblit=True, interactive=True, props=dict(alpha=0.5, facecolor='red'))
plt.show()

# show A and B
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


# ISF for various q for function of tau and q**2r
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

#free exponent
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

print(f"alpha = {alpha:.02f}")
print(f"X = {X:.02f}")

#Brownian motion
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
print(f"D = {D:.02f} µm²/s")



# D theory
D_theory = kB*T/(3*np.pi*mu*a) * 1e12
print(f"D theory = {D_theory:.02f} µm²/s")

# estimate the size of the particle
predicted_a = kB*T/(3*np.pi*mu*D) *1e12*1e6
print(f"Diameter = {predicted_a:.02f} µm")


# ballistic motion # commented out for now
# plt.figure(figsize=(8,6))
# plt.plot(qs, params[:,2], 'o')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$q\,(\mu m^{-1})$')
# plt.ylabel(r'Characteristic time $\tau_c (s)$')
# plt.axvspan(qs[iqmin], qs[iqmax], color=(0.9,0.9,0.9))

# v = np.exp(-leastsq(
#     lambda p, q, td: p - np.log(np.abs(q)) - np.log(np.abs(td)),
#     [-1],
#     args=(qs[iqmin:iqmax], params[iqmin:iqmax,2])
#     )[0][0])
# plt.plot([qs[iqmin], qs[iqmax]], 1/(v*np.array([qs[iqmin], qs[iqmax]])), '-r')
# print(f"v = {v:.02f} µm²/s")