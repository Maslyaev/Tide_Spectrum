#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Саш, у меня тут вопрос. В сетки частот, полученной из ДПФ у меня получается так, что некоторые частоты неплохо соответствуют приливам 
(например, 12.414 весьма неплохо соответствует 12.4206 от М2). Но вот с некоторыми приливами вроде 
'''
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.signal as signal
from netCDF4 import Dataset
import math

def Process_point(val_ts, expected_tide_period = 12.4):
    Domain = Spatial_Domain(val_ts)
    spectrum = Domain.Get_Amplitudes()
    eps = 0.03

    tide_period = expected_tide_period; tide_amplitude = 0
    for wave in spectrum:
        amplitude = wave[0]; period = wave[1]
        if abs(period - expected_tide_period) < eps:
            tide_period = expected_tide_period; tide_amplitude = amplitude
            print('difference between periods:', period - expected_tide_period, 'amplitude:', amplitude)            
            break
    return tide_amplitude
        

def Apply_Spectral_To_Matrix(Data, expected_tide_period = 12.4):
    '''
    
    main function of the framework: the first dimension of the Data must be time
    
    '''
    result_matrix = []
    for i in range(Data[0].shape[0]):
        res_row = []
        for j in range(Data[0].shape[1]):
            print('processing cell i: %4d, j:%4d; expected period: %3.2f' % (i, j, expected_tide_period))
            temp = Process_point(Data[:, i, j], expected_tide_period = expected_tide_period)
            res_row.append(temp)
        result_matrix.append(res_row)
    return result_matrix


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
        self.ts_transformed.real = self.ts_transformed.real / self.ts.shape[0]
        self.ts_transformed.imag = self.ts_transformed.imag / self.ts.shape[0]
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
            tide_period = 1/self.frequencies[np.argmax(self.power_spectrum)] 
            tide_amplitude = np.max(list(map(math.sqrt, self.power_spectrum)))
        return tide_period, tide_amplitude


def Create_variable(file, name, var_format, dimensions):
    real_part = file.createVariable(name + '_real_part', var_format, dimensions)
    imaginary_part = file.createVariable(name + '_imaginary_part', var_format, dimensions)
    return [real_part, imaginary_part]


def Create_netCDF(data_dict, file_name = 'noname.nc', file_description = '', example_netCDF = None):
    nc_file = Dataset('twst.nc', 'w', format = 'NETCDF4')
    nc_file.description = file_description
    
    nc_file.createDimension(dimname = 'x_grid_T', size = data[0].shape[0])
    nc_file.createDimension(dimname = 'y_grid_T', size = data[0].shape[1])

    x_grid_T = nc_file.createVariable('x_grid_T', 'f4', ('x_grid_T',))
    y_grid_T = nc_file.createVariable('y_grid_T', 'f4', ('y_grid_T',))
    harmonic_names = data_dict.keys()
    netCDF_variables = [Create_variable(nc_file, x, 'f4', ('x_grid_T', 'y_grid_T')) for x in harmonic_names]
    print(type(netCDF_variables), type(netCDF_variables[0]))
    netCDF_variables = [item for sublist in netCDF_variables for item in sublist]

    x_range =  np.linspace(0, data.shape[0]-1, data[0].shape[0])
    y_range =  np.linspace(0, data.shape[1]-1, data[0].shape[1]) 
    x_grid_T[:] = x_range
    y_grid_T[:] = y_range
    
    for variable in netCDF_variables:
        name = variable.name
        if 'real' in name:
            print(data_dict[name.replace('_real_part', '')].real.shape, variable[:, :].shape)
            variable[:, :] = data_dict[name.replace('_real_part', '')].real
        elif 'imaginary' in name:
            variable[:, :] = data_dict[name.replace('_imaginary_part', '')].imag
    
    nc_file.close()
    
    
if __name__ == "__main__":
    data = np.load('ssh_july.npy')
    Res_solar_diurnal = np.array(Apply_Spectral_To_Matrix(data, 24.0)).reshape(data[0].shape)
    Res_lunar_diurnal_K1 = np.array(Apply_Spectral_To_Matrix(data, 23.96)).reshape(data[0].shape)
    Res_lunar_diurnal_O1 = np.array(Apply_Spectral_To_Matrix(data, 25.8)).reshape(data[0].shape)
    Res_solar_semi_diurnal = np.array(Apply_Spectral_To_Matrix(data, 12.0)).reshape(data[0].shape)
    Res_lunar_semi_diurnal = np.array(Apply_Spectral_To_Matrix(data, 12.4)).reshape(data[0].shape)
    
    