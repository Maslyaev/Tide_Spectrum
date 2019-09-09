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
from shutil import copyfile
import os

import multiprocessing as mp

def Process_point(val_ts, d = 1.0, expected_tide_periods = [12.4], zeros_padding_number = 0):
    '''
    
    Calculate tide amplitudes for selected periods, using passed time-series;
    
    Parameters:    
        val_ts : float array, Contains time-series; 
        
        d : float, optional: default value of 1.0, Period between observations;    
        
        expected_tide_periods : list of array of floats, optional: default value of 12.4 (M2 - tide), Periods, for which the amplitudes are calculated;    
        
        zeros_padding_number : int, numbers of zeros, optional, Added padding for the array = val_ts.size * zeros_padding_number;
    
    Returns:
        amplitudes: : list of np.complex128, amplitudes for the expected_tide_periods
    
    '''
    
    Domain = Spatial_Domain(val_ts, d = d, only_spec_periods=True, periods=expected_tide_periods, zeros_padding_number = zeros_padding_number)
    spectrum = Domain.Get_Amplitudes()
    eps = 0.05
    
    amplitudes = []
    for expected_tide_period in expected_tide_periods:
        tide_amplitude = 0 #; tide_period = expected_tide_period
        for wave in spectrum:
            amplitude = wave[0]; period = wave[1]
            if abs(period - expected_tide_period) < eps:
                tide_amplitude = amplitude #tide_period = expected_tide_period;           
                break
        amplitudes.append(tide_amplitude)       
    return amplitudes

def dict_to_array(dictionary, shape):
    res = np.empty(shape)
    for idx, val in dictionary.items():
        val = np.array(val)
        res[idx[0], idx[1], :] = val[:]
    return res

def worker(indexes, Data, out_q, d, expected_tide_periods, zeros_padding_number, bathy):
    outdict = {}
    for index in indexes:
        print('processing cell i: %4d, j:%4d' % (index[0], index[1]))
        outdict[index] = Process_point(Data[:, index[0], index[1]], d = d, expected_tide_periods = expected_tide_periods, 
               zeros_padding_number = zeros_padding_number) if bathy[index] else [0j] * len(expected_tide_periods)
    out_q.put(outdict)

def Apply_Spectral_To_Matrix(Data, bathy = None, d = 1.0, expected_tide_periods = [12.4], zeros_padding_number = 0, nprocs = 2):
    '''
    
    Calculation of amplitudes of tidal waves, using data about SSH;
    
    Parameters:
        Data : float numpy matrix with 3 dimensions: time, x, y, Contains data about SSH for the selected area;
        
        bathy : numpy array of 0 and 1, optional: default values: 1, 
        Contains data about the conditions of the points if they are land (value of 0) or in the sea (value of 1). Recommended for speedup;
        
        d : float, optional: default value of 1.0, Period between observations;    
        
        zeros_padding_number : int, numbers of zeros, optional, Added padding for the array = val_ts.size * zeros_padding_number;
    
    Returns:
        result_matrix : np.complex128 numpy matrix of shape (Data.shape[1], Data.shape[2], len(expected_tide_periods)), containing amplitudes of tide;
    
    '''
    
    Data = Add_zeros_padding(Data, zeros_padding_number)

    try:
        _ = bathy.shape
    except AttributeError:
        bathy = np.ones(Data[0, :, :].shape)
    
    result_matrix = np.empty(Data[0].shape)
    iteridxs = np.array(list(np.ndenumerate(Data[0, :, :])))[:, 0]
    print(iteridxs.shape, iteridxs[-1])
    
    out_q = mp.Queue()
    chunksize = int(math.ceil(iteridxs.shape[0] / float(nprocs)))    
    procs = []

    for i in range(nprocs):
        p = mp.Process(
                target=worker,
                args=(iteridxs[chunksize*i : chunksize*(i+1)], Data, out_q, d, 
                               expected_tide_periods, zeros_padding_number, bathy))
        procs.append(p)
        p.start()

    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())    
        
    for p in procs:
        p.join()

    result_matrix = dict_to_array(resultdict, (Data.shape[1], Data.shape[2], len(expected_tide_periods)))
    return result_matrix


def Add_zeros_padding(Data, zeros_padding_number):
    '''
    Concatenate Data.shape[0] x zeros_padding_number number of zeros to the data matrix for improved amplitude calculation accuracy. Reduced the framework performance
    '''    

    zeros = np.zeros((Data.shape[0] * zeros_padding_number, Data.shape[1], Data.shape[2]))
    print('added zeros:', zeros.shape, 'base matrix:', Data.shape)
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
    '''
    Calculation of amplitudes for selected set of frequencies;
    
    Parameters:
        samples : numpy array of float64, containing series of values;
        
        freq : numpy array of float64, contating frequences for amplitudes;
        
        sample_freq : float64, sample frequency, inverse to the sample period;
        
    Returns:
        freq : numpy array of float64, contating frequences for amplitudes;
        
        X : numpy array of complex128, contatining complex amplitudes for selected frequencies;
    '''
    
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
    def __init__(self, time_series, d = 1.0, only_spec_periods = False, periods = [12.4], zeros_padding_number = 0):
        '''
        
        '''
        self.ts = np.array(time_series, dtype = np.float64)
        if only_spec_periods:
            freq = 1/np.array(periods)
            self.frequencies, self.ts_transformed = FT_single_freq(self.ts, freq, sample_freq = 1.0)
            self.ts_transformed = self.ts_transformed * (1 + zeros_padding_number)
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


def Create_netCDF(data_dict, data, file_name = 'noname.nc', file_description = '', example_netCDF = None):
    
    '''
    
    Write netCDF file with the calculated tide amplitudes; 
    
    Parameters:
        data_dict : dictionary with structure: key - index of tide (e.g. 'P1_Elevation_harmonic'), value - matrix of tide amplitude;
        
        data : numpy marix, Data input for the framework, used only for shape parameters;
        
        file_name : string, output file name;
        
        file_description : string, description for output netCDF file;
    
    '''
    
    nc_file = Dataset(file_name, 'w', format = 'NETCDF4')
    nc_file.description = file_description
    
    try:
        for dname, the_dim in example_netCDF.dimensions.iteritems():
            print(dname, len(the_dim))
            nc_file.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
    except AttributeError:
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
    
def Any_key_in_string(dictionary, string):
    for key, value in dictionary.items():
        if key in string:
            return True
    return False

def Create_netCDF_as_copy(data_dict, data, file_name = 'noname.nc', example_netCDF = None):
    
    '''
    
    Write netCDF file with the calculated tide amplitudes; 
    
    Parameters:
        data_dict : dictionary with structure: key - index of tide (e.g. 'P1_Elevation_harmonic'), value - matrix of tide amplitude;
        
        data : numpy marix, Data input for the framework, used only for shape parameters;
        
        file_name : string, output file name;    
        
    '''
    
    if os.path.isfile(file_name):
        print('Using existing target file:', file_name)
    else:
        print('Copying into new file', file_name)    
        copyfile(example_netCDF, file_name) 
    
    nc_file = Dataset(file_name, 'a', format = 'NETCDF4')

    print(data_dict.keys())
    for var_name in nc_file.variables:
        name = var_name
        if not Any_key_in_string(data_dict, name): 
            continue
        if 'real' in name:
            print(data_dict[name.replace('_real_part', '')].real.shape, nc_file.variables[var_name][:, :].shape)
            nc_file.variables[var_name][:, :] = data_dict[name.replace('_real_part', '')].real
        elif 'imaginary' in name:
            nc_file.variables[var_name][:, :] = data_dict[name.replace('_imaginary_part', '')].imag
    nc_file.close()
    
    
    