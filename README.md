# Code repository for the E1 Dynamics of Complex Fluids by Kogen Nobori

## Overview

Please refer to `DDM_main.py` for the main script for the analysis. This `DDM_main.py` conducts batch analysis of videos in a folder, and it is supplemented by `core.py` and `util.py`. `util.py` contains many fundamental classes and functions such as fitting with least_squares regression or RANSAC algorithm used in the code. `core.py` includes the core calculation processes in the analysis. The Python scripts include references where relevant.

## Details of the code

### `DDM_main.py`
#### Summary
By specifying the folder name, the code batch conducts Differential Dynamic Microscopy(DDM) analysis for the AVI videos in the folder. Options `ballistic`, `save`,  `no_p_size`, `additional_plots` can specify whether you are analysing ballistic motion, if you want to save results in a csv file, if you are analysing with mixed particles, and if you want additional plots for better visualisations. If `no_p_size` is specified as 2, the code runs $\tau_c$ against q and relevant calculations for the first particle and the second particle individually.

#### What are the main changes from the original code repository?
- The code is written in Python files(.py) rather than Jupyter Notebook(.ipynb) as the original repository had.
-  The code implements batch prediction and the analysis scripts are split into `DDM_main.py`, `core.py`, and `util.py` for better readability and extensibility.
-  The code also implements the formula by Likhachev for the temperature and pressure dependence of the water viscosity[1].
-  It implements correlation function for the solution with two different sizes. It can be expanded to three particles as well.
-  The calculated values such as predicted diameters or meta data of the video and the sample are stored with `pandas.DataFrame`.

### `core.py`
#### Summary
This script contains `core({arguments})` function that conducts free exponent fitting, Brown motion fitting, free exponent fitting with RANSAC, using functions and classes defined in `util.py`. The function returns the resulting values, which are passed to variables in `main.py`.

#### What are the main changes from the original code repository?
- `core({arguments})` conducts processes in the diameter prediction from the characteristic decay time and wavevector. It improves the readability and interpretability of the `main.py` by bring together the repeated processes in the analysis.
- It provides summary plot of various ways of fitting $\tau_c$ against q.

### `util.py`
#### Summary
This script contains various classes and functions used in the code, such as the classes and functions to calculate ISF, functions for fitting ISF, fitting ISF for solutions with particles of two different sizes, free exponent and Brownian fit, free exponent fit with RANSAC algorithm. Implementations of Brownian fitting (by forcing the free exponent to 2) with RANSAC algorithm or Bayesian RANSAC fitting using [the code](https://github.com/gsubramani/robust-nonlinear-regression) by Guru Subramani were tried but were not completed in time.

#### Errors in the original code repository
In the code in the original repository by samuelmcdermott, y axis of the plots produced by the code followed by "Show the results of the fit function $f(\tau,q)$ of $q$ for various $\tau$ and function of $\tau$ for various $q$." is written as $f(q,\tau)$, and the y axis of the plots by the code after "The corresponding ISF for various $q$ functions of $\tau$ or, to show the Brownian rescaling, function of $q^2\tau$." is written as $ISF(q,\tau)$. However, after looking into what they are actually plotting, I believe that they are reversed mistakenly and the former should be $ISF(q,\tau)$ and the latter should be $f(q,\tau)$. This can be updated to avoid future complications.

#### What are the main changes from the original code repository?
- `isf_fitting2` function defines a fitting function for solutions with particles of two different sizes. It conducts fitting by fitting the five parameters (instead of three for single particle) in the ISF model function for two particles. The resulting `params` have five parameters where `params[:,2]` and `params[:,3]` correspond to the characteristic decay time of the first and second particle against various wavevector q, respectively (Equation 8 in the report). Therefore, you can fit `params[:,2]` against q to get the diffusion coefficient of the first particle, and `params[:,3]` against q to get one for the second particle.  `params[:,4]` corresponds to the constant $C$ that describes the contributions from the two particles to the correlation function. `params[:,0]` and  `params[:,2]` correspond to optic/static structures and camera noise respectively(Equation 2 in the report). The author also tried to implement bounding parameters in the ISF fitting using `scipy.optimize.least_squares`, but it caused unknown errors and was not completed in time.
- functions `fr_exp_fit()`, `brwon_d()`, `fit_ransac()` fits  $\tau_c$ against q in the log-log scale. You can specify which column in `params` contain $\tau_c$ in the arguments, so this code can be used for analysis of mixed particles by fitting $\tau_c$ against q individually for different particles, as discussed above. They also have the option to neglect negative values in $\tau_c$ during fitting. `fit_ransac()` cannot function if the negative values are not neglected as it causes error.



# References in this README.md
[1]. Likhachev, E. R. Dependence of water viscosity on temperature and pressure. Tech. Phys. 48, 514–515 (2003).

