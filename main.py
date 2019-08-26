#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.signal as signal

def Periods_Decreasing_Order(amplitudes, freq, part = 'real'):
    if part == 'real':
        periods_sorted = [1/freq[x] for x in sorted(range(len(amplitudes.real)), key=lambda k: amplitudes.real[k])]
    elif part == 'imag':
        periods_sorted = [1/freq[x] for x in sorted(range(len(amplitudes.imag)), key=lambda k: amplitudes.imag[k])]
    else:
        return None
    periods_sorted.remove(np.inf)
    periods_sorted.reverse()
    return np.array(periods_sorted)
    
class Spatial_Domain:
    def __init__(self, time_series):
        
        '''
        
        point - time-series for one spatial point
        
        '''
        
        self.ts = time_series
        self.fourier_transform()
        
    def fourier_transform(self, print_mode = False):
        self.ts_transformed = fft.rfft(self.ts)
        self.frequencies = fft.rfftfreq(self.ts.size, d = 1.0) #/signal_rate
        self.periods_real = Periods_Decreasing_Order(self.ts_transformed, self.frequencies, part = 'real')
        self.periods_imag = Periods_Decreasing_Order(self.ts_transformed, self.frequencies, part = 'imag')
        if print_mode:
            print('real:', self.periods_real)
            print('imag:', self.periods_imag)
        
        
    def Plot_Result(self, x_axis = 'periods', transf_type = 'periodogram', size = 100):
        if x_axis == 'periods':
            if transf_type == 'periodogram':
                plt.plot(1/self.frequencies, self.power_spectrum)     # self.frequencies
            else:
                plt.plot(1/self.frequencies, self.ts_transformed.real, color = 'r')
        elif x_axis == 'frequencies':
            if transf_type == 'periodogram':
                plt.plot(self.frequencies, self.power_spectrum)     # self.frequencies
            else:
                plt.plot(self.frequencies, self.ts_transformed.real, color = 'r')
            
            #plt.plot(1/self.frequencies, self.ts_transformed.imag, color = 'b')            
            
    def Periodogram(self):
        fs = 1/3600
        f, Pxx_den = signal.periodogram(self.ts, fs, scaling = 'spectrum')
        plt.semilogy(f, Pxx_den)
        self.power_spectrum = Pxx_den
        self.frequencies = f
        plt.show()
    
    def Get_Amplitudes(self):
        return list(zip(np.flip(np.sort(self.ts_transformed)), self.periods_real))
    
    def Max_Power(self):
        try:
            tide_period = 1/Data_point.frequencies[np.argmax(Data_point.ts_transformed)] 
            tide_amplitude = np.max(list(map(math.sqrt, Data_point.ts_transformed)))
        except NameError:
            tide_period = 1/Data_point.frequencies[np.argmax(Data_point.power_spectrum)] 
            tide_amplitude = np.max(list(map(math.sqrt, Data_point.power_spectrum)))
        return tide_period, tide_amplitude