import os.path 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.interpolate import interp1d
from time import time as time
from time import strftime
from datetime import date
from matplotlib.figure import Figure
from scipy.signal import find_peaks, peak_widths
from scipy.signal import hilbert


from scipy import optimize
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size':16})








def gaussian(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset
def linear_func(x,m,c):
    return m*x +c

def quad_func(x,a,b,c):
    return a*x**2 + b*x +c 


def sech_squared(x,a,b,c):
    return a*(1/np.cosh(b*x))+c

def FROG_trace_folder(folder_path,save_folder_path, Relationship = None):

    temporal_widths_gauss =[]
    temporal_widths_sech =[]
    GDDs=[]
    
    folder = os.listdir(folder_path)
    

    for x in range(len(folder)):
        file_path = os.path.join(folder_path, folder[x])
        print('I am computing file',folder[x])
        #print(folder[x].split('_')[2])
        #GDD = int(folder[x].split('_')[4])
        #GDDs.append(GDD)
        #print(GDD)
        if folder[x] == '.DS_Store':   # removal of pesky MacOS .DS_Store files. Thye should be remove through terminal but this is a final catch all 
            continue 

        if folder[x].split('_')[-1] == '2d.txt':
            continue

        name = os.path.splitext(folder[x])[0]
        
        print(name)
        # Match numbers that are not part of a word (not next to letters)
        numbers = re.findall(r"(?<![A-Za-z])\d+(?![A-Za-z])", name)

        if numbers:
            number = max(map(int, numbers))
            print('The number is',number) 
            GDD = number
            GDDs.append(GDD) # 5000
        else:
            print("No standalone number found.")
        
       
       
        

       


        loaded_file = np.loadtxt(file_path)

        time = loaded_file[:,0]
        amplitude = loaded_file[:,1]


        analytic_signal = hilbert(amplitude)
        envelope = np.abs(analytic_signal)


        halfwidth =int(len(amplitude)/2)

        print(halfwidth)

        N = len(amplitude)
        center = N // 2
        

        start = center - int(halfwidth)
        end = center + int(halfwidth)
        signal_center = amplitude[start:end]
        envelope_center = envelope[start:end]
        time_center = time[start:end]


        peaks,peak_heights = find_peaks(signal_center,distance =10,prominence= 1e4)
        
        #print(len(peaks))

        A0= 2000*np.max(envelope_center)
        mu0 = time_center[np.argmax(envelope_center)]
        sigma0 = (time_center[-1] - time_center[0]) / 10
        offset0 = np.min(envelope_center)
        p0 = [A0, mu0, sigma0, offset0]
        

        params,params_covariance = curve_fit(gaussian, time_center[peaks], signal_center[peaks], p0=p0,maxfev =1000)
        params1,params_covariance1 = curve_fit(sech_squared, time_center[peaks], signal_center[peaks], p0=[A0,0.1,1])

        envelope_fit = (gaussian(time_center,(params[0]),params[1],params[2],params[3]))/np.max(gaussian(time_center,(params[0]),params[1],params[2],params[3]))
        envelope_fit_sech = (sech_squared(time_center,params1[0],params1[1],params1[2]))/np.max(sech_squared(time_center,params1[0],params1[1],params1[2]))
        time_fit = time_center[peaks]

        envelope_fit = envelope_fit - envelope_fit[0]
        envelope_fit_sech = envelope_fit_sech - envelope_fit_sech[0]


        fwhm = FWHM_2(time_center,envelope_fit)
        fwhm_2 = FWHM_2(time_center, envelope_fit_sech)

        #FROG_FWHM = fwhm/(np.sqrt(2))
        FROG_FWHM = fwhm/np.sqrt(2)
        #FROG_FWHM_2 = fwhm_2/(np.sqrt(2))
        FROG_FWHM_2 = fwhm_2/1.543
        temporal_widths_gauss.append(FROG_FWHM)
        temporal_widths_sech.append(FROG_FWHM_2)

        print('The estimated TL temporal width based on the above manipulation is {0:.3f}fs'.format(FROG_FWHM))

        plt.figure(figsize=(15,8))
        plt.plot(time, amplitude, color ='magenta', alpha = 0.8, label = 'SHG IAC 1d Trace')
        plt.plot(time_center,gaussian(time_center,(params[0]),params[1],params[2],params[3]), color='darkblue', label ='Gaussian Fit to SHG IAC peaks, Temp. Width ={0:.3f}fs'.format(FROG_FWHM))
        #plt.scatter(time_center[peaks], signal_center[peaks], color='red')
        plt.plot(time_center,sech_squared(time_center,params1[0],params1[1],params1[2]), color='darkorange', label ='$Sech^2$ Fit to SHG IAC peaks, Temp. Width ={0:.3f}fs'.format(FROG_FWHM_2))
        plt.xlabel("Time / fs")
        plt.ylabel("Arb. Intensity")
        plt.legend()
        title = 'SHG_IAC_trace_fit_GDD_'+str(GDD)+'.png'
        save_folder =os.path.join(save_folder_path,title)
        plt.title('SHG IAC Trace Fit for'+folder[x]+'')
        plt.savefig(save_folder)
        plt.show()
        plt.close()

   
    
    if len(GDDs) >1: 
        params, params_covariance = curve_fit(linear_func, GDDs, temporal_widths_gauss, p0=[1,0],maxfev =1000)
        #params1, params_covariance2 = curve_fit(quad_func,GDDs,temporal_widths_gauss, p0 =[1,0,0], maxfev= 1000)
        params2, params_covariance2 = curve_fit(linear_func, GDDs, temporal_widths_sech, p0=[1,0],maxfev =1000)

    #print(GDDs)
    GDDs =np.array(GDDs)

    fig = Figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(GDDs, temporal_widths_gauss, color ='red', label = 'IAC Gaussian Fit Recovered Data Points')
    ax.scatter(GDDs, temporal_widths_sech, color ='blue', label = 'IAC $sech^2$ Fit Recovered Data Points')
    ax.set_title('Temporal Width Applied GDD Relationship')
    ax.set_xlabel("Software Applied GDD ($fs^2$)")
    ax.set_ylabel("Temporal Width (fs)")
    
    if len(GDDs) >1 :
        ax.plot(GDDs,linear_func(GDDs, params2[0], params2[1]), color ='blue', label = '$sech^2$ LOBF => y= {0:.3f}x + {1:.3f}'.format(params2[0],params2[1]))
        ax.plot(GDDs,linear_func(GDDs, params[0], params[1]), color ='red', label = 'Guassian LOBF => y= {0:.3f}x + {1:.3f}'.format(params[0],params[1]))
    #plt.plot(GDDs,quad_func(GDDs,*params1), color ='red', label ='Quadratic LOBF')
    ax.legend()
    
    title = 'Time_GDD_Relationship_GDD_recovered_'+(date.today()).strftime("_%d_%m_%y")+'.png'
    save_folder =os.path.join(save_folder_path,title)
    
        
        

    text_title = 'Time_GDD_Relationship_GDD_recovered_'+(date.today()).strftime("_%d_%m_%y")+'.txt'
    save_name_2 =os.path.join(save_folder_path,text_title)

    #np.savetxt(save_name_2, np.c_[GDDs, temporal_widths_gauss, temporal_widths_sech],delimiter='\t')
    np.savetxt(save_name_2, np.c_[ temporal_widths_gauss, temporal_widths_sech],delimiter='\t')
    print(GDDs)
    print(GDDs)
    print('Gaussian', temporal_widths_gauss)
    print('sech',temporal_widths_sech)

    if Relationship == True: 
        return fig 
    else:
        return None       

def FROG_trace(file, GDD):

    loaded_file = np.loadtxt(file)

    time = loaded_file[:,0]
    amplitude = loaded_file[:,1]


    analytic_signal = hilbert(amplitude)
    envelope = np.abs(analytic_signal)
    



    N = len(amplitude)
    center = N // 2
    halfwidth =int(len(amplitude)/2)

    start = center - int(0.9*halfwidth)
    end = center + int(0.9*halfwidth)
    signal_center = amplitude[start:end]
    envelope_center = envelope[start:end]
    time_center = time[start:end]

    peaks,peak_heights = find_peaks(signal_center,distance =10,prominence= 1e4)
        
    #print(len(peaks))

    A0= 2000*np.max(envelope_center)
    mu0 = time_center[np.argmax(envelope_center)]
    sigma0 = (time_center[-1] - time_center[0]) / 10
    offset0 = np.min(envelope_center)
    p0 = [A0, mu0, sigma0, offset0]
        

    params,params_covariance = curve_fit(gaussian, time_center[peaks], signal_center[peaks], p0=p0,maxfev =1000)
    params1,params_covariance1 = curve_fit(sech_squared, time_center[peaks], signal_center[peaks], p0=[A0,0.1,1])

    envelope_fit = (gaussian(time_center,(params[0]),params[1],params[2],params[3]))/np.max(gaussian(time_center,(params[0]),params[1],params[2],params[3]))
    envelope_fit_sech = (sech_squared(time_center,params1[0],params1[1],params1[2]))/np.max(sech_squared(time_center,params1[0],params1[1],params1[2]))
    time_fit = time_center[peaks]

    envelope_fit = envelope_fit - envelope_fit[0]
    envelope_fit_sech = envelope_fit_sech - envelope_fit_sech[0]


    fwhm = FWHM_2(time_center,envelope_fit)
    fwhm_2 = FWHM_2(time_center, envelope_fit_sech)

    #FROG_FWHM = fwhm/(np.sqrt(2))
    FROG_FWHM = fwhm*0.648
    #FROG_FWHM_2 = fwhm_2/(np.sqrt(2))
    FROG_FWHM_2 = fwhm_2*0.648
  

    print('The estimated TL temporal width based on the above manipulation is {0:.3f}fs'.format(FROG_FWHM))

    plt.figure(figsize=(15,8))
    plt.plot(time, amplitude, color ='magenta', label = 'SHG IAC Trace')
    #plt.scatter(time_center[peaks], signal_center[peaks], color='red')
    #plt.plot(time_center,gaussian(time_center,(params[0]),params[1],params[2],params[3]), color='darkblue', label ='Gaussian Fit to IAC peaks, Temp. Width ={0:.3f}fs'.format(FROG_FWHM))
    plt.plot(time_center,sech_squared(time_center,params1[0],params1[1],params1[2]), color='darkorange', label ='$Sech^2$ Fit to IAC peaks, Temp. Width ={0:.3f}fs'.format(FROG_FWHM_2))
    plt.xlabel("Time \ fs ")
    plt.ylabel(" Intensity \ Arb. Units")
    plt.legend()
    plt.xlim(-250,250)
    title = 'SHG_IAC_trace_fit_GDD_'+str(GDD)+'.png'
    plt.title('SHG IAC trace fit, GDD = '+str(GDD)+'$fs^2$')
    plt.savefig(title)
    plt.show()
    
    
    return FROG_FWHM

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
    #else:
        #return np.nan
    
    # Right side
    right_region = y[peak:]
    if np.any(right_region <= half_max):
        y1 = np.where(right_region <= half_max)[0][0] + peak
        y0 = y1 - 1
        x_right = x[y1] + (x[y0] - x[y1]) * (half_max - y[y1]) / (y[y0] - y[y1])
    #else:
        #return np.nan
    
    fwhm = x_right -x_left
    return fwhm
