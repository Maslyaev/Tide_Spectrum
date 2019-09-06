#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:35:43 2019

@author: mike_ubuntu
"""


import numpy as np
from netCDF4 import Dataset
import Tide_Spectrum as Tides
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

def Plot_Harmonic(index, nc_file_object):
    Heatmap(np.sqrt(nc_file_object.variables[index+'_Elevation_harmonic_real_part'][:]**2 
            + nc_file_object.variables[index+'_Elevation_harmonic_imaginary_part'][:]**2))

if __name__ == "__main__":
    bathy_file = Dataset('bathy_meter_mask.nc', 'r', format='NETCDF4')
    bathy = bathy_file.variables["Bathymetry"][:].data
    bathy_file.close()
    
    data = np.load('ssh_july.npy')[:, :, :]
    
    periods = [24.0, -24.0, 23.96, -23.96, 25.74, -25.74, 12.0, -12.0, 12.4, -12.4]
    harmonics = Tides.Apply_Spectral_To_Matrix(data, bathy = bathy, expected_tide_periods = periods, zeros_padding_number=0)
    harmonics_matrix = np.array(harmonics).reshape((data[0].shape[0], data[0].shape[1], len(periods)))

    Tides.Create_netCDF_as_copy({'P1_Elevation_harmonic' : abs(harmonics_matrix[:, :, 0].real) + abs(harmonics_matrix[:, :, 1].real) + (abs(harmonics_matrix[:, :, 0].imag) + abs(harmonics_matrix[:, :, 1].imag))*1j,
                   'K1_Elevation_harmonic' : abs(harmonics_matrix[:, :, 2].real) + abs(harmonics_matrix[:, :, 3].real) + (abs(harmonics_matrix[:, :, 2].imag) + abs(harmonics_matrix[:, :, 3].imag))*1j,
                   'O1_Elevation_harmonic' : abs(harmonics_matrix[:, :, 4].real) + abs(harmonics_matrix[:, :, 5].real) + (abs(harmonics_matrix[:, :, 4].imag) + abs(harmonics_matrix[:, :, 5].imag))*1j,
                   'S2_Elevation_harmonic' : abs(harmonics_matrix[:, :, 6].real) + abs(harmonics_matrix[:, :, 7].real) + (abs(harmonics_matrix[:, :, 6].imag) + abs(harmonics_matrix[:, :, 7].imag))*1j,
                   'M2_Elevation_harmonic' : abs(harmonics_matrix[:, :, 8].real) + abs(harmonics_matrix[:, :, 9].real) + (abs(harmonics_matrix[:, :, 8].imag) + abs(harmonics_matrix[:, :, 9].imag))*1j}, 
        data = data, file_name = 'tides_new.nc', example_netCDF='tides_20130715.nc')
    
    try:
        Tides_file = Dataset('tides_new.nc', 'r', format='NETCDF4')
        Harmonic_Idx = 'M2' # Select harmonic index, supported ones: P1, K1, O1, S2, M2
        Plot_Harmonic(Harmonic_Idx, Tides_file)
    except RuntimeError:
        print('Visual representation of the results is not supported on this machine: download netCDF for further work')
    
    