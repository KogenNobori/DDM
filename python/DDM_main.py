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

# onselect
def onselect(xmin, xmax):
    global iqmin, iqmax
    iqmin= np.searchsorted(qs, xmin)
    iqmax = np.searchsorted(qs, xmax)
    print('\n')
    print(f"Selected range: {qs[iqmin]:.2f} to {qs[iqmax]:.2f}")

def select_valid_q(qs, params, param_col):
    # span selector
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(qs, params[:, param_col], 'o') #Choose the column of 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$q\ [\mu m^{-1}$]')
    ax.set_ylabel(r'Characteristic time $\tau_c\ [s]$')
    ax.set_title('Click and drag to select a valid range of q values')

    span = SpanSelector(ax, onselect, 'horizontal', useblit=True, interactive=True, props=dict(alpha=0.5, facecolor='red'))
    plt.show()


################################################
#########        change below       ############
################################################
folder_name = '/Volumes/T7 Shield/e1_stand1/data/Concentrationdiff2mium/rt' # Input the folder to analyse
out_dir = '/Users/noboriwatarugen/e1/ddm_github_forked/python/csvs/' # remember to add / in the end

pixelSize = 100/844.341 #40x (in micrometre)

pointsPerDecade = 30
maxNCouples = 3 #10 for fast evaluation, 300 for accurate analysis

kB = 1.38e-23
T = 273.15+23.1 # measured room temperatur 
mu = 8.9e-4 # change this value later

ballistic = False # Input true if it is ballistic
save = True # Input true if you want to save the results in the csv file
no_p_size = 1 #Input the number different kinds of particles.
additional_plots = True


################################################
#############  change above ####################
################################################






###### Choose the correlation function###

if ballistic: # if ballistic is true, use the correlation function for the ballistic motion
    LogISF = lambda p, dts: np.log(p[0] * (1-(np.sin(dts/p[2])/(dts/p[2]))) + p[1]) 

elif no_p_size==1: # correlation function for the Brownian motion
    LogISF = lambda p, dts: np.log(np.maximum(p[0] * (1 - np.exp(-dts / p[2])) + p[1], 1e-10))

elif no_p_size==2:
    LogISF = lambda p, dts: np.log(p[0] * (1 - (p[4]*np.exp(-dts / p[2])+(1-p[4])*np.exp(-dts / p[3]))) + p[1])




p_size_dict = {'2':2, '2p12':2.12, '0_977u':0.977, '1_73u':1.73,'2u':2,'1p73':1.73,'01':0.1,'1':1,'03':0.3,'173':1.73 } # https://www.sejuku.net/blog/24122


####Storing the results in dataframe 
data_columns = ['Particle size','Dilution','Frame rate','Number of frames','qmin', 'qmax','alpha','X','D (Brownian)','D theory (Brownian)','Predicted Diameter (Brownian)', 'Velocity(for ballistic)']
df = pd.DataFrame(columns = data_columns) # https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.html

# df.to_csv(out_dir+'test.csv')


########### Path ########### https://note.nkmk.me/python-pathlib-iterdir-glob/
folder_path = Path(folder_name) 
video_file_paths = list(folder_path.glob('*.avi'))  

print('\n')
print('Video file paths:', list(map(str, video_file_paths))) # https://note.nkmk.me/python-map-usage/
print('\n')
print('Number of files to analise:', len(video_file_paths))
print('\n')
parent_dir_name = folder_path.name # https://note.nkmk.me/python-pathlib-name-suffix-parent/
# print(parent_dir_name)

##########################

for video_file in video_file_paths:
    ##### Extracting information from the filename ####
    # file_path = Path(video_file) #https://note.nkmk.me/python-pathlib-name-suffix-parent/#google_vignette #https://docs.python.org/ja/3/library/pathlib.html#pathlib.PurePath.name #https://docs.python.org/3.15/library/pathlib.html#
    print('File being analysed:', video_file.name)

    # Dilution
    dil = int(re.findall('1-(.*?)_',video_file.name)[0]) # https://note.nkmk.me/python-str-extract/#research-refindall
    print('Dilution:', dil)


    # Particle size
    p_size = re.findall('([0-9].??)m',video_file.name) # https://note.nkmk.me/python-str-extract/#research-refindall # why do .?? and .*? behave differently?
    print('Found sequences:', p_size)
    print('\n')



    if no_p_size==1 and len(p_size)==1 and p_size[0] in p_size_dict.keys(): # If the code only catches one string and it matches one of the keys,
        a = p_size_dict[p_size[0]] #change this to the diameter of the particle in metres

    else:
        if no_p_size==1:
            print('Particle size not found. Please input the particle size in micro metre:')
            a = float(input())*1e-6
        elif no_p_size==2:
            print('Please input the first particle size in micro metre:')
            a1 = float(input())*1e-6
            print('Please input the second particle size in micro metre:')
            a2 = float(input())*1e-6

    print('\n')
    print('Particle size in micro metre:', a*1e6,'µm')
    print('\n')


    ########### Reading the data###########
    stack = util.ImageStack(video_file)

    ########## calculate isf ###########
    idts = util.logSpaced(len(stack), pointsPerDecade)
    dts = idts/stack.fps
    ISF = util.calculate_isf(stack, idts, maxNCouples)

    # calculate the scale of wavevectors
    qs = 2*np.pi/(2*ISF.shape[-1]*pixelSize) * np.arange(ISF.shape[-1])
    tmax = -1

    if additional_plots:
        # display the isf 
        util.plot_isf(dts, qs, ISF)

    if no_p_size==1:

        # perform the ISF fitting

        params, matrixfit = util.isf_fitting(LogISF, dts, ISF, tmax)


        # span selector
        select_valid_q(qs, params, 2)


        # show A and B
        if additional_plots:
            # show A and B
            util.plot_AB(qs, params, iqmin, iqmax)

            # result of the fit function
            util.plot_result_fit(dts, qs, matrixfit, ISF, iqmin, iqmax, tmax)

            # ISF for various q for function of tau and q**2r
            util.plot_isf_4_q(dts, qs, params, ISF, iqmin, iqmax, tmax)

        #free exponent
        X, alpha = util.fr_exp_fit(qs, params, iqmin, iqmax)

        print('\n')
        print('Free exponent fitting results:')
        print(f"alpha = {alpha:.02f}")
        print(f"X = {X:.02f}")
        print('\n')

        D = 'None'
        D_theory = 'None'
        Diameter = 'None'
        predicted_a = 'None'

        if not ballistic:
            # Brownian motion
            D = util.brown_d(qs, params, iqmin, iqmax)
            print('Brownian results:')
            print(f"D = {D:.02f} µm²/s")


            # D theory
            D_theory = kB*T/(3*np.pi*mu*a) * 1e12
            # print('raw D theory', D_theory)
            print(f"D theory = {D_theory:.02f} µm²/s")

            # estimate the size of the particle
            predicted_a = kB*T/(3*np.pi*mu*D) *1e12*1e6
            print(f"Precicted Diameter = {predicted_a:.02f} µm")
            print(f"Manufactured Diameter = {a*1e6:.02f} µm")
            print('\n')
            print('\n')


        # ballistic motion 
        v = 'None'
        if ballistic:
            v = util.ballistic(qs, params, iqmin, iqmax)
            print('Ballistic results:')
            print(f"v = {v:.02f} µm²/s")
            print('\n')
            print('\n')

    if no_p_size==2:
            # perform the ISF fitting
            params, matrixfit = util.isf_fitting2(LogISF, dts, ISF, tmax)
    
    
            # span selector
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
            if additional_plots:
                # show A and B
                util.plot_AB(qs, params, iqmin, iqmax)
    
                # result of the fit function
                util.plot_result_fit(dts, qs, matrixfit, ISF, iqmin, iqmax, tmax)
    
                # ISF for various q for function of tau and q**2r
                util.plot_isf_4_q(dts, qs, params, ISF, iqmin, iqmax, tmax)
    
            #free exponent
            X, alpha = util.fr_exp_fit(qs, params, iqmin, iqmax)
    
            print('\n')
            print('Free exponent fitting results:')
            print(f"alpha = {alpha:.02f}")
            print(f"X = {X:.02f}")
            print('\n')
    
            D = 'None'
            D_theory = 'None'
            predicted_a = 'None'
    
            
            # Brownian motion
            D = util.brown_d(qs, params, iqmin, iqmax)
            print('Brownian results:')
            print(f"D = {D:.02f} µm²/s")


            # D theory
            D_theory = kB*T/(3*np.pi*mu*a) * 1e12
            # print('raw D theory', D_theory)
            print(f"D theory = {D_theory:.02f} µm²/s")

            # estimate the size of the particle
            predicted_a = kB*T/(3*np.pi*mu*D) *1e12*1e6
            print(f"Precicted Diameter = {predicted_a:.02f} µm")
            print(f"Manufactured Diameter = {a*1e6:.02f} µm")
            print('\n')
            print('\n')






    #### Storing data # https://note.nkmk.me/python-pandas-assign-append/
    df_add = pd.DataFrame([[a, dil, stack.fps, stack.frame_count, qs[iqmin], qs[iqmax], alpha, X, D, D_theory, predicted_a, v]], index = [video_file.name], columns=df.columns)

    df = pd.concat([df, df_add])


print('Resulting data')
print(df)

if save:
    df.to_csv(out_dir+'mnc{}_{}_output.csv'.format(maxNCouples, parent_dir_name)) # https://note.nkmk.me/python-pandas-to-csv/#google_vignette