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
# Using scipy
import scipy

# import original modules
import util
import core

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

def select_valid_q(qs, params, param_col, label=None):
    # span selector
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(qs, params[:, param_col], 'o') #Choose the column of 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$q\ [\mu m^{-1}$]')
    ax.set_ylabel(r'Characteristic time $\tau_c\ [s]$')
    ax.set_title('Click and drag to select a valid range of q values. Label: {}'.format(label))

    span = SpanSelector(ax, onselect, 'horizontal', useblit=True, interactive=True, props=dict(alpha=0.5, facecolor='red'))
    plt.show()


################################################
#########        change below       ############
################################################
folder_name = '/Volumes/T7 Shield/e1_stand1/data/mixed_with_new_conc_calibration/mixed_videos/2/27' # Input the folder to analyse
out_dir = '/Users/noboriwatarugen/e1/ddm_github_forked/python/csvs/' # remember to add / in the end

lense = 40 

if lense ==40:
    pixelSize = 100/844.341 #40x (in micrometre)

pointsPerDecade = 30
maxNCouples = 10 #10 for fast evaluation, 300 for accurate analysis

kB = 1.38e-23
T = 273.15+23.1 # measured room temperature

# Reference: Likhachev, E. R. Dependence of water viscosity on temperature and pressure. Tech. Phys. 48, 514–515 (2003).

E = 4.753*1e3 #J/mol
eta_0 = 2.4055*1e-5 #Pa s
theta = 139.7 #K
a_const = 4.42 * 1e-4 #bar-1
b_const = 9.565* 1e-4 *1e3 #J/(mol bar)
c_const = 1.24e-2 #K/bar


p = 1 #bar
T = 273.15+23.1 #K

eta = eta_0*np.exp(a_const*p+(E-b_const*p)/(scipy.constants.R*(T-theta-c_const*p)))


mu = eta # viscosity

ballistic = False # Input true if it is ballistic
save = True # Input true if you want to save the results in the csv file
no_p_size = 2 #Input the number different kinds of particles.
# neglect_negative =True # Whether to neglect negative values in characteristic time when fitting tauc vs q
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
data_columns = ['lense','stack.fps', 'stack.frame_count','a_i', 'a_ii', 'dil_i', 'dil_ii', 'D_theory_i', 'D_theory_ii','qs[iqmin1]', 'qs[iqmax1]', 'alpha1','X1', 'D1', 'predicted_a1',  'alphar1','Xr1', 'predicted_a_ransac_free1', 'v1', 'qs[iqmin2]', 
                'qs[iqmax2]', 
                'alpha2', 
                'X2', 
                'D2',  
                'predicted_a2', 
                'alphar2',
                'Xr2',
                'predicted_a_ransac_free2',
                'v2'
                ] 
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



    # Particle size
    p_size = re.findall('([0-9].??)m',video_file.name) # https://note.nkmk.me/python-str-extract/#research-refindall # why do .?? and .*? behave differently?
    print('Found sequences:', p_size)
    print('\n')



    if no_p_size==1 and len(p_size)==1 and p_size[0] in p_size_dict.keys(): # If the code only catches one string and it matches one of the keys,
        a = p_size_dict[p_size[0]] #change this to the diameter of the particle in metres
        # Dilution
        # dil = int(re.findall('1-(.*?)_',video_file.name)[0]) # https://note.nkmk.me/python-str-extract/#research-refindall

        print('Input the dilution:')
        dil = int(input())

        print('\n')
        print('Particle size:', a*1e6,'µm')
        print('Dilution:', dil)
        print('\n')

    else:
        if no_p_size==1:
            print('Particle size not found. Please input the particle size in micro metre:')
            a = float(input())*1e-6

            print('Input the dilution:')
            dil = int(input())

            print('\n')
            print('Particle size:', a*1e6,'µm')
            print('Dilution:', dil)
            print('\n')

        elif no_p_size==2:
            print('Please input the first particle size in micro metre:')
            a_i = float(input())*1e-6

            print('Input the dilution of the first particle:')
            dil_i = int(input())


            print('Please input the second particle size in micro metre:')
            a_ii = float(input())*1e-6

            print('Input the dilution of the second particle:')
            dil_ii = int(input())

            print('\n')
            print('Particle size1:', a_i*1e6,'µm')
            print('Dilution1:', dil_i)
            print('Particle size2:', a_ii*1e6,'µm')
            print('Dilution2:', dil_ii)
            print('\n')

    


    ########### Reading the data###########
    stack = util.ImageStack(video_file)
    #Plotting the first frame
    # plt.imshow(stack[0])
    # plt.title('First frame')
    # plt.show()

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
        select_valid_q(qs, params, param_col=2) # Input the column index of params that is used for the fitting

        X, alpha, D, predicted_a, v = core.core(qs, dts, ISF, params, matrixfit, iqmin, iqmax, tmax, kB, T, mu, additional_plots, ballistic, no_p_size, param_col=2)

        D_theory = util.calc_d_theory(kB, T, mu, a)


        # #### Storing data # https://note.nkmk.me/python-pandas-assign-append/
        # df_add = pd.DataFrame([[lense, stack.fps, stack.frame_count,a, 'None', dil, 'None', qs[iqmin], qs[iqmax], alpha, X, D, D_theory, predicted_a, v, 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None']], index = [video_file.name], columns=df.columns)
    
        # df = pd.concat([df, df_add])
        # print('\n')
        # print("Current data:")
        # print(df)




        

    if no_p_size==2:
        # perform the ISF fitting
        params, matrixfit = util.isf_fitting2(LogISF, dts, ISF, tmax)

        # First particle
        print('Analysis of the first particle')
        label='First particle'
        # span selector
        select_valid_q(qs, params, param_col=2, label=label) # https://www.geeksforgeeks.org/python/keyword-and-positional-argument-in-python/

        iqmin1 = iqmin
        iqmax1 = iqmax

        X1, alpha1, D1, predicted_a1, Xr1, alphar1, predicted_a_ransac_free1, v1 = core.core(qs, dts, ISF, params, matrixfit, iqmin, iqmax, tmax, kB, T, mu, additional_plots, ballistic, no_p_size,param_col=2, label=label)

        # Second particle
        print('Analysis of the second particle')
        label='Second particle'
        # span selector
        select_valid_q(qs, params, param_col=3, label=label)

        iqmin2 = iqmin
        iqmax2 = iqmax

        X2, alpha2, D2, predicted_a2, Xr2, alphar2, predicted_a_ransac_free2, v2 = core.core(qs, dts, ISF, params, matrixfit, iqmin, iqmax, tmax, kB, T, mu, additional_plots, ballistic,  no_p_size, param_col=3, label=label)

        D_theory_i = util.calc_d_theory(kB, T, mu, a_i)
        D_theory_ii = util.calc_d_theory(kB, T, mu, a_ii)

        print('\n')
        print('Particle sizes:', a_i, a_ii)
        print('D_theory:', D_theory_i, D_theory_ii)
        print('Calculated D:', D1, D2)
        print('Predicted particle sizes:',predicted_a1, predicted_a2)
        print('Predicted particle sizes (RANSAC free exponent):',predicted_a_ransac_free1, predicted_a_ransac_free2)



        ### Storing data # https://note.nkmk.me/python-pandas-assign-append/
        df_add = pd.DataFrame([[lense,
                                 stack.fps, 
                                 stack.frame_count,
                                 a_i, 
                                 a_ii, 
                                 dil_i, 
                                 dil_ii, 
                                 D_theory_i, 
                                 D_theory_ii,
                                 qs[iqmin1], 
                                 qs[iqmax1], 
                                 alpha1,
                                 X1, 
                                 D1, 
                                 predicted_a1, 
                                 alphar1,
                                 Xr1,
                                 predicted_a_ransac_free1,
                                 v1, 
                                 qs[iqmin2], 
                                 qs[iqmax2], 
                                 alpha2, 
                                 X2, 
                                 D2,  
                                 predicted_a2, 
                                 alphar2,
                                 Xr2,
                                 predicted_a_ransac_free2,
                                 v2
                                 ]], 
                                 index = [video_file.name], 
                                 columns=df.columns)

        df = pd.concat([df, df_add])
        print('\n')
        print("Current data:")
        print(df)      





print('\n')
print('Resulting data')
print(df)

if save:
    df.to_csv(out_dir+'mnc{}_{}_output.csv'.format(maxNCouples, parent_dir_name)) # https://note.nkmk.me/python-pandas-to-csv/#google_vignette