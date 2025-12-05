
# The following code aims to simulate dispersion due to 2nd order optical phase (GDD) for a gaussian ultrafast laser pulse. 
# The code is quite basic and has been adapted from a larger .ipynb file that performed the same function, as such it may have references to 
# other unncessary bits of code.
# The aim of this was that certain sections would be applied to real laser spectra to simulate phase influenced dispersion in more complex systems. 


## Importing Libraries and Setting Font Size 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from statistics import pstdev
from scipy.fft import fftshift, ifft, fftfreq
import numpy as np
import math
from scipy.interpolate import interp1d
from time import time as time
from time import strftime
from datetime import date
from scipy.signal import ShortTimeFFT
from scipy.signal import find_peaks, peak_widths
from scipy.stats import norm
from IPython.display import display, Latex
import re 
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
plt.rcParams.update({'font.size':12})


# Defining Functions used in future calculations


def FWHM_2(x,y):
    peak =np.argmax(y)
    peak_val =y[peak]
    half_max = peak_val /2 

    # Left side
    left_region = y[:peak]
    if np.any(left_region <= half_max):
        y1 = np.where(left_region <= half_max)[0][-1]
        y0 = y1 + 1
        # linear interpolation for crossing
        x_left = x[y1] + (x[y0] - x[y1]) * (half_max - y[y1]) / (y[y0] - y[y1])
    else:
        return np.nan

    # Right side
    right_region = y[peak:]
    if np.any(right_region <= half_max):
        y1 = np.where(right_region <= half_max)[0][0] + peak
        y0 = y1 - 1
        x_right = x[y1] + (x[y0] - x[y1]) * (half_max - y[y1]) / (y[y0] - y[y1])
    else:
        return np.nan

    fwhm = x_right -x_left
    return fwhm

def quad_func(x,a,b,c,d,e):
    y = a*x**4 + b*x**3 + c*x**2 +d*x + e 
    return y    

def linear_func(x,m,c):
    y=m*x +c 
    return y


# Set out the global variables and simulated laser spectrum parameters


lambda0 = 800.0  # central wavelength (nm)
full_width_half = 63    # spectral FWHM (nm)
num_points = 200000     # number of point for the linspace 


# Create the gaussian laser spectrum in a linspace and define the intensity and amplitude

lambdas = np.linspace(lambda0 - 3*full_width_half, lambda0 + 3*full_width_half, num_points)  #raw wavelength domain linspace
sigma = full_width_half / (2*np.sqrt(2*np.log(2)))                                           # gaussian distribution function 
I_lambda = np.exp(-0.5 * ((lambdas - lambda0) / sigma)**2)                                   # create laser intensity from the distribution function 
A_lambda = np.sqrt(I_lambda / np.max(I_lambda))                                              # convert intensity to amplitude                       
I_lambda /= np.max(I_lambda)                                                                 # normalise the intensity 
A_lambda = np.sqrt(I_lambda / np.max(I_lambda))                                              # normalise the amplitude                           

# we then plot the laser spectrum in the wavelength domain

plt.figure(figsize=(8,5))
plt.plot(lambdas, I_lambda, color='blue')
plt.title("Simulated Ultrashort Laser Spectrum (Normalised)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Intensity (a.u.)")
plt.show()

# for the fft to time we require the spectrum in the angular frequency (omega) domain rather than wavelength. hence we convert. 


c = 2.998e8                                                                                 # define the speed of light in m/s
omega = (2*np.pi*c / (lambdas * 1e-9))/1e15                                                 #convert the wavelength linspace 
omega0 = (2*np.pi*c / (lambda0 * 1e-9))/1e15                                                # convert the central wavelength
Omega = omega - omega0                                                                      # all spectra must be represented as detuned from central frequency hence this line

                                                
                                             
idx = np.argsort(Omega)                                                                     # Sort the angular frequency in ascending order for Fourier compatibility
omegas, I_omega = Omega[idx], I_lambda[idx]                                                 # sort the intensity as well 

# for a check we then plot in the angular frequency domain 

plt.figure(figsize=(8,5))
plt.plot(omegas, I_omega, color='blue')
plt.title("Simulated Ultrashort Laser Spectrum (Normalised)")
plt.xlabel("Angular Frequency (1e15$rads^{-1})$")
plt.ylabel("Normalized Intensity (a.u.)")
plt.show()

# Next generate the phase function - in this case only GDD is present so we have a quadratic 

a2 = 885                                                                                    # GDD in s^2, adjust to desired value 
phase_data = -0.5*(a2 * (Omega)**2)                                                         # create the phase function based on the angular frequency linspace


# Next we required an evenly spaced grid in the frequency domain for the FFT 

delta_omega = (Omega.max()-Omega.min())/num_points                                          # define the number of angular frequency units per step
Omega_uniform = Omega.min() + np.arange(num_points) *delta_omega                            # create the omega grid
A_interp = interp1d(Omega, A_lambda, kind='cubic', fill_value=2000, bounds_error=False)     # sets the interpolation function for aligning the amplitude of the spectrum to the new grid 
phi_interp = interp1d(Omega, phase_data, kind='cubic', fill_value=2000, bounds_error=False) # repeats the above but for the phase function 
A_omega = A_interp(Omega_uniform)                                                           # perform the spectrum interpolation 
phi_omega = phi_interp(Omega_uniform)                                                       # perform the phase interpolation 

# We then create the omega domain electric field by complex mulitplying the interpolated amplitude with the phase ampltidue 
E_omega = A_omega * np.exp(1j * phi_omega)

# Finally perform the inverse FFT into the time domain using scipy.fftshift 

E_t = fftshift(ifft(fftshift(E_omega)))                                                     # perform the FFT (inverse)
domega = Omega_uniform[1] - Omega_uniform[0]                                                # re-find the frequency domain spacing 
dt = 2*np.pi / (num_points * domega)                                                        # use this to find the spacing of the time domain 
t = dt * np.arange(-num_points/2, num_points/2)                                             # create a linspace in time based off this spacing
I_t = np.abs(E_t)**2                                                                        # turn the new time domain electric field into an intensity 
I_t /= np.max(I_t)                                                                          # normalise the intensity 

# Then find the temporal FWHM of the resultant, print and then plot the result

fw = FWHM_2(t,I_t)                                                                          # apply the function from above
print('The FWHM of the resultant temporal pulse is {0:.3f}fs'.format(float(fw)))            # print the output nicely 

plt.figure(figsize=(10,4))                                                                  # plot the functional form of the applied phase mask
plt.plot(lambdas, phase_data, 'r')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Phase (rad)")
plt.title("Applied Phase Mask")
plt.grid(True)

plt.figure(figsize=(10,4))                                                                  # Plot the resultant temporal profile        
plt.plot(t, I_t, 'k', label ='FWWH ={0:.3f}'.format(fw))
plt.xlabel("Time (fs)")
plt.ylabel("Normalized Intensity")
plt.title("Temporal Pulse After Phase Mask")
plt.grid(True)
plt.xlim(-600,600)                                                                          # this limit might need adjusted for very broad pulses 
plt.tight_layout()
plt.show()


# Next create a relationship between applied GDD and resultant FWHM using a loop 

phases = np.linspace(0,800,1000)                                                            # create a linspace of phases to analyse, this can be customised and the lower bound > 0 at all times 

widths =[]                                                                                  #empty array 
for i in range(len(phases)):    

    a2 = phases[i]                                                                          #all maths in this function the same as before 
    phase_data = -0.5*(a2 * (Omega)**2)  # 
    phi_interp = interp1d(Omega, phase_data, kind='cubic', fill_value=2000, bounds_error=False)
    phi_omega = phi_interp(Omega_uniform)
    E_omega = A_omega * np.exp(1j * phi_omega)
    E_t = fftshift(ifft(fftshift(E_omega)))
    I_t = np.abs(E_t)**2
    I_t /= np.max(I_t)
    fwg = FWHM_2(t,I_t)
    widths.append(float(fwg))

widths =np.asarray(widths)                                                                   #force this into an array for nice plotting 

def gauss_fit(x,c):                                                                           # this function create the relationship for a perfect gaussian broadening with t_0=15.1899 set manually, this could  be globally set 
    return 15.1899*np.sqrt(1+((c*x)/(15.1899)**2)**2)

params,params_covariance = curve_fit(gauss_fit,phases, widths, p0=[1.5])                      # use optimize.curve_fit to create the fit to the data to extra c - ideally c=4ln(2) = 2.773

plt.figure(figsize=(10,6))                                                                   #plot the output of the loop with the data and the perfect gaussian fit also    
plt.plot(phases, widths,color ='blue', label ='Simulated Data')
plt.plot(phases,gauss_fit(phases,params[0]), color ='orange', label ='Fit, C ={0:.3f}'.format(params[0]))
plt.xlabel("GDD \ $fs^2$")
plt.ylabel("Temporal Width \ fs")
plt.title("Temporal Width (FWHM) as a function of simulated applied GDD")
plt.legend()
plt.show()

#Finally we can produce a broadening graph based on an interpolation to the data generated  via the loop above
interp_phase = PchipInterpolator(widths, phases)                                             #define the interpoalation to the phase data from the above plot. We could also extract this from the fit to the data but this will handle poorer quality fits better

times =[15.2,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]            # create a set of times from the TL limit t0 200fs in this case 
widths4=[]                                                                                   # empty arrays for saving values to 
intensities4=[]
temp_profiles4=[]
dispersion4=[]
for i in range(len(times)):                                                                  #loop repeats from above for every member of times[]

    target = times[i]                           
    gdd1 = interp_phase(target)                                                              #interpolate to the phase
    dispersion4.append(gdd1)                    
    a2 = gdd1                               
    phase_data = -0.5*(a2 *(Omega)**2)                                                       #create phase function based on the interpolation 
    phi_interp = interp1d(Omega, phase_data, kind='cubic', fill_value=2000, bounds_error=False)
    phi_omega = phi_interp(Omega_uniform)
    E_omega = A_omega * np.exp(1j *phi_omega)                                                 # set this as the electric field in the frequency domain
    E_t = fftshift(ifft(fftshift(E_omega)))                                                   #convert to time domain
    I_t = np.abs(E_t)**2
    I_t /= np.max(I_t)                                                                         # normalise
    gh4 =FWHM_2(t,I_t)                                                                         # calc FWHM
    widths4.append(float(gh4))                                                                 # save values for plotting
    intensities4.append(I_t)
    temp_profiles4.append(t)

#Finally, finally plot the outcome of the above loop

color = iter(plt.cm.rainbow(np.linspace(0, 1, len(widths4))))                                  # create a colour map for each of the different curves. This will handle very large numbers of samples (>10000)
cmap = plt.get_cmap('rainbow', len(widths4))

plt.figure(figsize=(20,13))
for i in range(len(intensities4)):                                                              #plot the curves looping over the colours 
    c = next(color)
    plt.plot(temp_profiles4[i],intensities4[i], color =c, label ='GDD ={0:.3f}$fs^2$  => FWHM = {1:.3f}fs'.format(dispersion4[i],widths4[i]) )
plt.xlabel("Time \ fs ")
plt.ylabel("Normalised Intensity \ Arb. Units ")
plt.title("Temporal Widths Correlated to Applied GDD")
plt.xlim(-400,400)
title ='Temporal_Widths_Correlation_with_prediction.png'
#plt.savefig(title)                                                                             # figure can be save for export as with all figures in this document 
plt.show()