#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:35:43 2019

@author: mike_ubuntu
"""

import sys
import numpy as np
from netCDF4 import Dataset
import Tide_Spectrum_parallel as Tides
import matplotlib.pyplot as plt

def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1))):
    y, x = np.meshgrid(np.linspace(area[0][0], area[0][1], Matrix.shape[0]), np.linspace(area[1][0], area[1][1], Matrix.shape[1]))
    fig, ax = plt.subplots()
    if interval:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=interval[0], vmax=interval[1])    
    else:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=min(-abs(np.max(Matrix)), -abs(np.min(Matrix))),
                          vmax=max(abs(np.max(Matrix)), abs(np.min(Matrix)))) 
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()

def Plot_Harmonic(index, nc_file_object, harmonic_name):
    Heatmap(np.sqrt(nc_file_object.variables[index + '_' + harmonic_name + '_harmonic_real_part'][:]**2 
            + nc_file_object.variables[index + '_' + harmonic_name + '_harmonic_real_part'][:]**2))
    
def Plot_Difference(index, nc_file_object1, nc_file_object2, harmonic_name):
    Heatmap(abs(np.sqrt(nc_file_object1.variables[index + '_' + harmonic_name + '_harmonic_real_part'][:]**2 
            + nc_file_object1.variables[index + '_' + harmonic_name + '_harmonic_imaginary_part'][:]**2) - 
        np.sqrt(nc_file_object2.variables[index + '_' + harmonic_name + '_harmonic_real_part'][:]**2 
            + nc_file_object2.variables[index + '_' + harmonic_name + '_harmonic_imaginary_part'][:]**2)))

def Process_Variable(Datafile_name, var_name, bathy, periods = [12.4], timespan = 240, res_filename = 'tides.nc', nprocs = 2):
    data = np.load(Datafile_name)
    if np.ndim(data) > 3:
        data = data[:, 0, :, :]
        print('Reduced dimensionality of data, using top-layer:', data)
    if data.shape[0] > timespan:
        data = data[:timespan, :, :]
    
    
    harmonics = Tides.Apply_Spectral_To_Matrix(data, bathy = bathy, expected_tide_periods = periods, zeros_padding_number=0, nprocs=nprocs)
    harmonics_matrix = np.array(harmonics).reshape((data[0].shape[0], data[0].shape[1], len(periods)))
    harmonic_type = var_name + '_harmonic'

    # Hardcoded tide periods; todo: more flexible setting of relations between indexes and matrices
    Tides.Create_netCDF_as_copy({'P1_' + harmonic_type : abs(harmonics_matrix[:, :, 0].real) + abs(harmonics_matrix[:, :, 1].real) + (abs(harmonics_matrix[:, :, 0].imag) + abs(harmonics_matrix[:, :, 1].imag))*1j,
                   'K1_' + harmonic_type : abs(harmonics_matrix[:, :, 2].real) + abs(harmonics_matrix[:, :, 3].real) + (abs(harmonics_matrix[:, :, 2].imag) + abs(harmonics_matrix[:, :, 3].imag))*1j,
                   'O1_' + harmonic_type : abs(harmonics_matrix[:, :, 4].real) + abs(harmonics_matrix[:, :, 5].real) + (abs(harmonics_matrix[:, :, 4].imag) + abs(harmonics_matrix[:, :, 5].imag))*1j,
                   'S2_' + harmonic_type : abs(harmonics_matrix[:, :, 6].real) + abs(harmonics_matrix[:, :, 7].real) + (abs(harmonics_matrix[:, :, 6].imag) + abs(harmonics_matrix[:, :, 7].imag))*1j,
                   'M2_' + harmonic_type : abs(harmonics_matrix[:, :, 8].real) + abs(harmonics_matrix[:, :, 9].real) + (abs(harmonics_matrix[:, :, 8].imag) + abs(harmonics_matrix[:, :, 9].imag))*1j,
                   'N2_' + harmonic_type : abs(harmonics_matrix[:, :, 10].real) + abs(harmonics_matrix[:, :, 11].real) + (abs(harmonics_matrix[:, :, 10].imag) + abs(harmonics_matrix[:, :, 11].imag))*1j,
                   'Q1_' + harmonic_type : abs(harmonics_matrix[:, :, 12].real) + abs(harmonics_matrix[:, :, 13].real) + (abs(harmonics_matrix[:, :, 12].imag) + abs(harmonics_matrix[:, :, 13].imag))*1j,
                   'K2_' + harmonic_type : abs(harmonics_matrix[:, :, 12].real) + abs(harmonics_matrix[:, :, 13].real) + (abs(harmonics_matrix[:, :, 12].imag) + abs(harmonics_matrix[:, :, 13].imag))*1j}, 
        data = data, file_name = 'tides_new.nc', example_netCDF='tides_20130715.nc')
    
    

if __name__ == "__main__":
    
    bathy_file = Dataset('bathy_meter_mask.nc', 'r', format='NETCDF4')
    bathy = bathy_file.variables["Bathymetry"][:].data 
    bathy_file.close()

    periods = [24.07, -24.07, 23.93, -23.93, 25.82, -25.82, 12.0, -12.0, 12.42, -12.42, 12.66, -12.66, 26.87, -26.87, 11.97, -11.97]

    filenames = ['ssh_july.npy', 'curr_V_july.npy', 'curr_U_july.npy']
    Var_Names = ['Elevation', 'Current_V', 'Current_U']
    
    print('sys.argv:', sys.argv[1])
    nprocs = int(sys.argv[1])
    
    for idx in range(len(Var_Names)):
        print('processing variable ', Var_Names[idx])
        Process_Variable(filenames[idx], Var_Names[idx], periods, res_filename='tides_full.nc', nprocs = nprocs)

    
#    try:
#        Tides_file = Dataset('tides_new.nc', 'r', format='NETCDF4')
#        Harmonic_Idx = 'M2' # Select harmonic index, supported ones: P1, K1, O1, S2, M2
#        Plot_Harmonic(Harmonic_Idx, Tides_file)
#    except RuntimeError:
#        print('Visual representation of the results is not supported on this machine: download netCDF for further work')
    
    