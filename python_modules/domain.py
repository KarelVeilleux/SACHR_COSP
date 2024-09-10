import sys;             sys.path.append('/home/veilleux/Projet/Projet_SACHR/analyses_and_figures/python_modules')
from   grid_projection  import read_gem_settings
from   grid_projection  import convert_rotlatlon_to_cartesian as latlon2indx

import numpy as np

##############################################################################################################
def construct_coord(x,y):
    coord = []
    N     = len(x)
    for n in range(N):
        coord.append((x[n],y[n]))
    return coord

##############################################################################################################
def set_domain(domain):
        if   domain == 'box11'            : grid = read_gem_settings('NAM-11m.nml');  xi=250; xf=575; yi=205; yf=490; d=0.1;
        elif domain == 'NAM11'            : grid = read_gem_settings('NAM-11m.nml');  xi=  0; xf=655; yi=  0; yf=655; d=0.1;
        elif domain == 'great_lakes_11'   : grid = read_gem_settings('NAM-11m.nml');  xi=315; xf=430; yi=262; yf=338; d=0.1; #Great Lakes
        elif domain == 'sonora_desert_11' : grid = read_gem_settings('NAM-11m.nml');  xi=130; xf=185; yi=140; yf=230; d=0.1; #Sonara desert
        elif domain == 'bermuda_azores_11': grid = read_gem_settings('NAM-11m.nml');  xi=530; xf=615; yi=145; yf=230; d=0.1; #Bermuda-Azores High
        elif domain == 'hudson_bay_11'    : grid = read_gem_settings('NAM-11m.nml');  xi=290; xf=410; yi=355; yf=480; d=0.1; #Hudson bay
        elif domain == 'pacific_sw_11'    : grid = read_gem_settings('NAM-11m.nml');  xi= 40; xf=100; yi=100; yf=200; d=0.1; #Pacific-SW
        elif domain == 'mexico_south_11'  : grid = read_gem_settings('NAM-11m.nml');  xi=230; xf=400; yi= 25; yf= 90; d=0.1; #Mexico-south
        elif domain == 'bc_coast_11'      : grid = read_gem_settings('NAM-11m.nml');  xi= 60; xf=190; yi=355; yf=480; d=0.1; #BC
        elif domain == 'ECAN2p5'          : print('Not implemented yet'); exit()
        else:                               grid = read_gem_settings(domain + '.nml'); xi = 0; yi = 0; di=0.1
        return grid, xi, xf, yi, yf, d

##############################################################################################################
def generate_domain_coord(domain,output_type='pair'):

    grid, xi, xf, yi, yf, d = set_domain(domain)
    seg1_i = np.arange(xi, xf+d, d); seg1_j = seg1_i * 0 + yi
    seg2_j = np.arange(yi, yf+d, d); seg2_i = seg2_j * 0 + xf
    seg3_i = np.arange(xf, xi-d,-d); seg3_j = seg3_i * 0 + yf
    seg4_j = np.arange(yf, yi-d,-d); seg4_i = seg4_j * 0 + xi
    i = np.concatenate((seg1_i, seg2_i, seg3_i, seg4_i))
    j = np.concatenate((seg1_j, seg2_j, seg3_j, seg4_j))
    lon, lat = latlon2indx(i  , j  , grid , 'free', 'index2lonlat')
    
    if output_type == 'pair':
        coord = construct_coord(lon,lat)
    else:
        coord = {'lon': lon, 'lat': lat}

    return coord

##############################################################################################################
def convert_latlon_to_domain_indices(track, domain):
    grid, xi, xf, yi, yf, d = set_domain(domain)
    track_i, track_j = latlon2indx(track['longitude'], track['latitude'], grid, 'free', 'lonlat2index')
    track_roundi     = np.round(track_i).astype(int)
    track_roundj     = np.round(track_j).astype(int)

    Nray = len(track_i)
    indx = {}
    indx['i'] = np.empty(Nray, dtype=int)
    indx['j'] = np.empty(Nray, dtype=int)
    for n in range(Nray):
        indx['j'][n] = track_roundi[n] - xi - 1
        indx['i'][n] = track_roundj[n] - yi - 1

    return indx

