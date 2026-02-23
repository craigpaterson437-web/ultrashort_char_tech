import os.path 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.figure import Figure
from time import time as time
from time import strftime
from datetime import date

plt.rcParams.update({'font.size':16})



def phase_mask_creation(generated_phase_file,TL_compensation_file,save_folder, plot = None):
    
    loaded_phase_file = np.loadtxt(generated_phase_file)

    phase_wave = loaded_phase_file[:,0]
    phase_amp = loaded_phase_file[:,1]

    comp_data_load = np.loadtxt(TL_compensation_file)
    comp_wave = comp_data_load[:,0]
    comp_amp = comp_data_load[:,1]

    phase_summation = comp_amp + phase_amp

    if plot == True: 
    
        plt.figure(figsize=(15,8))
        
        plt.plot(phase_wave,phase_amp, color ='blue', label ='Generated Mask') 
        plt.plot(comp_wave,comp_amp, color ='red', label ='Compansation Mask') 
        plt.plot(comp_wave,phase_summation, color ='magenta', label ='Resultant Summation Mask') 
        
        plt.xlabel("Wavelength (nm)")
        plt.ylabel(" Intensity (Arb. Units)")
        plt.title("Summary of Phase Mask Form")
        plt.legend()
        

        plt.show()

    f = generated_phase_file.split('/')[-1]
    GDD = str(f.split('_')[-2])
    title = ''+GDD+'_GDD_TL_summed_mask_'+(date.today()).strftime("_%d_%m_%y")+'.txt'
    save_name =os.path.join(save_folder,title)

    np.savetxt(save_name, np.c_[comp_wave, phase_summation],delimiter='\t')



def phase_mask_creation_folder(generated_phase_file_folder, TL_compensation_file, save_folder, plot = None):

    comp_data_load = np.loadtxt(TL_compensation_file)
    comp_wave = comp_data_load[:,0]
    comp_amp = comp_data_load[:,1]

    load_folder = os.listdir(generated_phase_file_folder)

    for x in range(len(load_folder)):

        file_path = os.path.join(generated_phase_file_folder, load_folder[x])
        #print('I am computing file',load_folder[x])

        loaded_phase_file = np.loadtxt(file_path, skiprows =9)

        phase_wave = loaded_phase_file[:,0]
        phase_amp = loaded_phase_file[:,1]

        phase_summation = comp_amp + phase_amp

        GDD = str(load_folder[x].split('_')[-2])
        title = ''+GDD+'_GDD_TL_summed_mask_'+(date.today()).strftime("_%d_%m_%y")+'.txt'

        if plot == True: 

            plt.figure(figsize=(25,8))
        
            plt.plot(phase_wave,phase_amp, color ='blue', label ='Generated Mask') 
            plt.plot(comp_wave,comp_amp, color ='red', label ='Compansation Mask') 
            plt.plot(comp_wave,phase_summation, color ='magenta', label ='Resultant Summation Mask') 
            plt.xlabel("Wavelength (nm)")
            plt.ylabel(" Intensity (Arb. Units)")
            plt.title(title)
            plt.legend()
            plt.show()

        
        save_name =os.path.join(save_folder,title)

        np.savetxt(save_name, np.c_[comp_wave, phase_summation],delimiter='\t')



def create_trans_mask(lambda_c, lambda_width, slm_size, bandwidth, save_folder): 

    centre_lambda = lambda_c
    width = lambda_width

    delta_lambda = bandwidth

    low = centre_lambda- (width/2)
    up = (width/2) + centre_lambda
    
    pixels = np.linspace(1,slm_size, slm_size)
    lambdas = np.linspace(low, up, slm_size)



    low_bound = (np.mean(lambdas)) - (delta_lambda/2)
    upper_bound = (np.mean(lambdas)) + (delta_lambda/2) 


    intensity =[]


    for x in range(len(lambdas)):

        if lambdas[x] < low_bound:

            intensity.append(0)

        if  low_bound <= lambdas[x] <= upper_bound: 

            intensity.append(1)

        if  lambdas[x] > upper_bound:

            intensity.append(0)


   

    title ='Trans_mask_'+str(bandwidth)+'_nm.txt'
    save_name =os.path.join(save_folder,title)

    np.savetxt(save_name, np.c_[lambdas, intensity],delimiter='\t')

    
def make_the_gaussian(lambda_c, slm_size, bandwidth, save_folder, Display = None):

    fundamental_file =r'/Users/craigpaterson/Library/CloudStorage/OneDrive-UniversityofGlasgow/5th Year/Project Data/redspex.txt'
    loaded_file = np.loadtxt(fundamental_file)

    waves = loaded_file[:,0]
    inten = loaded_file[:,1]
    low = waves[0]
    up = waves[len(waves)-1]

    lambda0 = lambda_c  # central wavelength (nm)a
    full_width_half = bandwidth  # spectral FWHM (nm)
    num_points = slm_size


  

    #lambdas1 = np.linspace(lambda0 - 2*full_width_half, lambda0 + 2*full_width_half, num_points)
    lambdas1 = np.linspace(low,up,num_points)
    sigma = full_width_half / (2*np.sqrt(2*np.log(2)))
    I_lambda = np.exp(-0.5 * ((lambdas1 - lambda0) / sigma)**2)
    A_lambda = np.sqrt(I_lambda / np.max(I_lambda))
    title ='Trans_mask_gaussian_'+str(bandwidth)+'_nm.txt'
    # --- Normalize ---
    I_lambda /= np.max(I_lambda)
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(lambdas1, I_lambda, color ='red', label ='Gaussian Transmission Mask - {0}$nm$'.format(bandwidth))
    ax.plot(waves, inten/np.max(inten), color ='blue', label = 'Laser Fundamental Spectrum')
    ax.set_title(title)
    ax.set_xlabel("Wavelength / $nm$")
    ax.set_ylabel(" Normalised Arb. Intensity")


    save_name =os.path.join(save_folder,title)

    np.savetxt(save_name, np.c_[lambdas1, I_lambda],delimiter='\t')
    if Display == True:
        return fig
    else:
        return None
   
    
