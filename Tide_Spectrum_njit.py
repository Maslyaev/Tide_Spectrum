#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

'''
import math

import numpy as np
import numba as nb 
import numpy.fft as fft

import matplotlib.pyplot as plt
from netCDF4 import Dataset


def Process_point(val_ts, expected_tide_periods = [12.4]):
    Domain = Spatial_Domain(val_ts, only_spec_periods=True, periods=expected_tide_periods)
    spectrum = Domain.Get_Amplitudes()
    eps = 0.05
    
    amplitudes = []
    for expected_tide_period in expected_tide_periods:
        tide_period = expected_tide_period; tide_amplitude = 0
        for wave in spectrum:
            amplitude = wave[0]; period = wave[1]
            if abs(period - expected_tide_period) < eps:
                tide_period = expected_tide_period; tide_amplitude = amplitude
                #print('difference between periods:', period, 'amplitude:', amplitude)            
                break
        amplitudes.append(tide_amplitude)       
    return amplitudes
        

def Apply_Spectral_To_Matrix(Data, bathy, expected_tide_periods = [12.4], zeros_padding_number = 0):
    '''
    
    main function of the framework: the first dimension of the Data must be time
    
    '''
    
    Data = Add_zeros_padding(Data, zeros_padding_number)

    result_matrix = []
    for i in range(Data[0].shape[0]):
        res_row = []
        for j in range(Data[0].shape[1]):
            print('processing cell i: %4d, j:%4d' % (i, j)) #; expected period: %3.2f , expected_tide_periods
            if bathy[i, j]:
                temp = Process_point(Data[:, i, j], expected_tide_periods = expected_tide_periods)
                res_row.append(temp)
            else:
                temp = [0j] * len(expected_tide_periods)
                res_row.append(temp)
        result_matrix.append(res_row)
    return result_matrix


def Add_zeros_padding(Data, zeros_padding_number):
    zeros = np.zeros((Data.shape[0] * zeros_padding_number, Data.shape[1], Data.shape[2]))
    return np.concatenate((Data, zeros), axis = 0)

def Periods_Decreasing_Order(amplitudes, freq, part = 'real'):
    if part == 'real':
        periods_sorted = [1/freq[x] for x in sorted(range(len(amplitudes.real)), key=lambda k: amplitudes.real[k])]
    elif part == 'imag':
        periods_sorted = [1/freq[x] for x in sorted(range(len(amplitudes.imag)), key=lambda k: amplitudes.imag[k])]
    else:
        return None
    if np.inf in periods_sorted:
        periods_sorted.remove(np.inf)
    periods_sorted.reverse()
    return np.array(periods_sorted)

@nb.jit(nb.types.Tuple((nb.float64[:], nb.complex128[:]))(nb.float64[:], nb.float64[:], nb.float64), nopython=True)
def FT_single_freq(samples, freq, sample_freq = 1.0):
    k = np.full(freq.shape[0], samples.size, dtype = np.float64)*freq*np.full(freq.shape[0], sample_freq, dtype = np.float64) #_unrounded 
    X = np.zeros(freq.size, dtype = np.complex128)
    for k_idx in np.arange(k.shape[0]):
        exp_pow = 1j*np.complex128(-2*np.pi*np.round(k[k_idx])/samples.size)
        X_val = 0j
        for n_idx in np.arange(samples.size):
            X_val = X_val + np.complex128(samples[n_idx])*np.exp(exp_pow*np.complex128(n_idx))
        X[k_idx] = X_val/samples.shape[0]
    return freq, X


    
class Spatial_Domain:
    def __init__(self, time_series, d = 1.0, only_spec_periods = False, periods = [12.4]):
        
        '''
        
        point - time-series for one spatial point
        
        '''
        
        self.ts = np.array(time_series, dtype = np.float64)
        if only_spec_periods:
            freq = 1/np.array(periods)
            #print(type(self.ts), type(freq))
            self.frequencies, self.ts_transformed = FT_single_freq(self.ts, freq, sample_freq = 1.0)
            self.periods_real = Periods_Decreasing_Order(self.ts_transformed, self.frequencies, part = 'real')
            self.periods_imag = Periods_Decreasing_Order(self.ts_transformed, self.frequencies, part = 'imag')            
        else:
            self.fourier_transform(d = 1.0)
            
    
    def fourier_transform(self, d = 1.0, print_mode = False):
        self.ts_transformed = fft.fft(self.ts)
        self.ts_transformed.real = self.ts_transformed.real / self.ts.shape[0]
        self.ts_transformed.imag = self.ts_transformed.imag / self.ts.shape[0]
        self.frequencies = fft.fftfreq(self.ts.size, d = d) #/signal_rate
        self.periods_real = Periods_Decreasing_Order(self.ts_transformed, self.frequencies, part = 'real')
        self.periods_imag = Periods_Decreasing_Order(self.ts_transformed, self.frequencies, part = 'imag')
        if print_mode:
            print('real:', self.periods_real)
            print('imag:', self.periods_imag)
        
        
    def Plot_Result(self, x_axis = 'periods', size = 100):
        if x_axis == 'periods':
            plt.plot(1/self.frequencies, self.ts_transformed.real, color = 'r')
        elif x_axis == 'frequencies':
            plt.plot(self.frequencies, self.ts_transformed.real, color = 'r')

    
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
    nc_file = Dataset(file_name, 'w', format = 'NETCDF4')
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
    bathy_file = Dataset('bathy_meter_mask.nc', 'r', format='NETCDF4')
    bathy = bathy_file.variables["Bathymetry"][:].data
    bathy_file.close()
    
    data = np.load('ssh_july.npy')[:, :, :]
    
    periods = [24.0, -24.0, 23.96, -23.96, 25.74, -25.74, 12.0, -12.0, 12.4, -12.4]
    harmonics = Apply_Spectral_To_Matrix(data, bathy, periods, zeros_padding_number=1)
    harmonics_matrix = np.array(harmonics).reshape((data[0].shape[0], data[0].shape[1], len(periods)))

    Create_netCDF({'P1_Elevation_harmonic' : abs(harmonics_matrix[:, :, 0]) + abs(harmonics_matrix[:, :, 1]),
                   'K1_Elevation_harmonic' : abs(harmonics_matrix[:, :, 2]) + abs(harmonics_matrix[:, :, 3]),
                   'O1_Elevation_harmonic' : abs(harmonics_matrix[:, :, 4]) + abs(harmonics_matrix[:, :, 5]),
                   'S2_Elevation_harmonic' : abs(harmonics_matrix[:, :, 6]) + abs(harmonics_matrix[:, :, 7]),
                   'M2_Elevation_harmonic' : abs(harmonics_matrix[:, :, 8]) + abs(harmonics_matrix[:, :, 9])}, file_name = 'tides_serial.nc')
    
    
    