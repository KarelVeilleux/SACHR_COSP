import numpy as np
NaN = np.nan

import sys;             sys.path.append('/home/veilleux/Projet/Projet_SACHR/analyses_and_figures/python_modules')
from   domain           import set_domain
from   grid_projection  import convert_rotlatlon_to_cartesian as latlon2indx

import matplotlib.pyplot as plt

#################################################################################################
def format2D(data1D, longitude, latitude, domain, fillvalue=NaN):
    
    # Check lon lat are 1D
    Ndimlon = len(longitude.shape)
    Ndimlat = len(latitude.shape)
    if Ndimlon != 1 or Ndimlat != 1:
        print('ERROR: longitude and latitude must be 1 dimensional')
        exit()

    # Check data1D dimension
    Ndim = len(data1D.shape)
    if not (Ndim == 1 or Ndim == 2) :
        print('ERROR: Data to format must be 1 dimensional (horizontal) or 2 dimensional (horizontal + vertical)')
        exit()
    elif Ndim == 1:
        # Note here we are adding a single vertical to be able to treat the 1D and the (1+1)D case together: 1D --> (1+1)D
        data1D = data1D[np.newaxis,:]
    elif Ndim == 2:
        # Making sure that the dimension order are (vertical, horizontal)
        if data1D.shape[1] != Ndimlon:
            data1D = np.transpose(data1D)
 
    
    # Extracting domain info (note: x --> lon, y --> lat)
    grid, xi, xf, yi, yf, d = set_domain(domain)
    Nlat                    = yf - yi
    Nlon                    = xf - xi 
    Nvertical               = data1D.shape[0]

    # Setting data2D values
    data2D = np.ones((Nvertical, Nlat, Nlon) ) * fillvalue  # Structure: vertical, lat, lon (as many other data)
    for n in range(len(longitude)):
        i, j          = latlon2indx(longitude[n], latitude[n], grid, 'free','lonlat2index')
        i             = np.round(i).astype(int)
        j             = np.round(j).astype(int)
        
        if i >= xi and i < xf and j >= yi and j < yf :
            data2D[:,j,i] = data1D[:,n]

    if Nvertical == 1: return data2D[0,:,:]
    else:              return data2D
 
