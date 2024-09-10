import matplotlib.pyplot as     plt
import matplotlib        as     mpl
import numpy             as     np
import pandas            as     pd
import xarray            as     xr
import cartopy.feature   as cfeature
import cartopy.crs       as ccrs
import yaml
import sys
import os
import netCDF4
import warnings

from   pylab             import cm
from   scipy             import ndimage
warnings.filterwarnings("ignore")

sys.path.append('/home/veilleux/Projet/Projet_SACHR/analyses_and_figures/python_modules')
from domain         import generate_domain_coord
from domain         import convert_latlon_to_domain_indices
from satellite      import extract_satellite_track


#################################################################################################################
##########                                           FUNCTIONS                                       ############
#################################################################################################################
def create_dataframe(dirlist, YYYYMM):
    # Create a dataframe containg only the files of YYYYMM 
    YYYY =     str(YYYYMM[0:4])
    MM   = int(str(YYYYMM[4:6]))

    filelist   = dirlist + '/' + YYYY + '.txt'
    df         = pd.read_csv(filelist,delimiter='\s+', header=None)
    df.columns = ['file', 'ndata', 'ti', 'tf', 'date', 'MM', 'date_gem', 't_gem']
    df         = df[   np.isin(df['MM'], MM)  ].reset_index(drop=True)
    return df

def convert_calipso_data_in_2D(ncfile, varname, coord_domain, ni, nj):

    track   = extract_satellite_track(ncfile, coord_domain, 'calipso')
    indices = convert_latlon_to_domain_indices(track, 'NAM11')

    nc      = netCDF4.Dataset(ncfile,'r')
    data    = nc[varname][track['index']]

    data_sum  = np.ones((655,655)) * 0
    data_n    = np.ones((655,655)) * 0
    for i in range(len(indices['i'])): 
        I = indices['i'][i]
        J = indices['j'][i]
        data_sum[I, J] = data_sum[I, J] + data[i]
        data_n[  I, J] = data_n[  I, J] + 1
    return data_sum/data_n

def monthly_mean(df,filepath_CALIPSO,filepath_GEM,filepath_COSP):
    ########################################################
    ####          Initializing data arrays              ####
    ########################################################
    datasets             = ['calipso','gem','cosp']
    layers               = ['t','h','m','l']

    n    = {}
    sum_data  = {}
    data = {}
    for dataset in datasets:
        sum_data  [dataset] = {}
        data [dataset] = {}
        for layer in layers:
            n            [layer]      = np.zeros((655,655))
            sum_data [dataset][layer] = np.zeros((655,655))
            data[dataset][layer]      = np.zeros((655,655))
    common_mask = {}
    ########################################################
    ########################################################

    for i in range(len(df)):
        t_gem        = df['t_gem'   ][i]
        YYYY_gem     = str(df['date_gem'][i])[0:4]
        YYYYMMDD_gem = str(df['date_gem'][i])
        YYYYMM_gem   = str(df['date_gem'][i])[0:6]
        date         = str(df['date'    ][i])
        MM           = str(df['date_gem'][i])[4:6]
        CALIPSO_file = df['file'][i].split('/')[8]

        CALIPSO_nc   = filepath_CALIPSO.replace('YYYY',YYYY_gem) +'/'+ CALIPSO_file
        GEM5_nc      = filepath_GEM.replace('YYYYMMDD',YYYYMMDD_gem).replace('YYYYMM',YYYYMM_gem)
        COSP_nc      = filepath_COSP.replace('YYYYMMDD',str(date)[0:8]).replace('HH',str(date)[8:10]).replace('YYYYMM',str(date)[0:6])
        
        if  os.path.exists(CALIPSO_nc) and os.path.exists(GEM5_nc) and os.path.exists(COSP_nc):
            print('works')
            # CALIPSO, GEM5 & COSP datasets
            ds_CALIPSO = netCDF4.Dataset(CALIPSO_nc, 'r')
            ds_GEM5    = netCDF4.Dataset(GEM5_nc   , 'r')
            ds_COSP    = netCDF4.Dataset(COSP_nc   , 'r')

            ########################################################
            ######                 CALIPSO                    ######
            ########################################################
            coord_NAM11   = generate_domain_coord('NAM11')
            data['calipso']['t']   = convert_calipso_data_in_2D(CALIPSO_nc,  'Tot_cloud_cover', coord_NAM11, 655, 655)
            data['calipso']['h']   = convert_calipso_data_in_2D(CALIPSO_nc, 'High_cloud_cover', coord_NAM11, 655, 655)
            data['calipso']['m']   = convert_calipso_data_in_2D(CALIPSO_nc,  'Mid_cloud_cover', coord_NAM11, 655, 655)
            data['calipso']['l']   = convert_calipso_data_in_2D(CALIPSO_nc,  'Low_cloud_cover', coord_NAM11, 655, 655)
            #CALIPSO mask        
            calipso_mask_t   = np.where(np.isnan(data['calipso']['t']), 0, 1)
            calipso_mask_h   = np.where(np.isnan(data['calipso']['h']), 0, 1)
            calipso_mask_m   = np.where(np.isnan(data['calipso']['m']), 0, 1)
            calipso_mask_l   = np.where(np.isnan(data['calipso']['l']), 0, 1)

            data['calipso']['t'][np.isnan(data['calipso']['t'])] = 0
            data['calipso']['h'][np.isnan(data['calipso']['h'])] = 0
            data['calipso']['m'][np.isnan(data['calipso']['m'])] = 0
            data['calipso']['l'][np.isnan(data['calipso']['l'])] = 0

            calipso_mask_82deg = np.where(np.array(ds_GEM5['lat']) >= 82, np.nan,1)

            data['calipso']['t'] = data['calipso']['t'] * calipso_mask_82deg
            data['calipso']['h'] = data['calipso']['h'] * calipso_mask_82deg
            data['calipso']['m'] = data['calipso']['m'] * calipso_mask_82deg
            data['calipso']['l'] = data['calipso']['l'] * calipso_mask_82deg

            ########################################################
            ######                    GEM5                    ######
            ########################################################
            data['gem']['t']    = ds_GEM5['TCCM'][t_gem]
            data['gem']['h']    = ds_GEM5['TZHM'][t_gem]
            data['gem']['m']    = ds_GEM5['TZMM'][t_gem]
            data['gem']['l']    = ds_GEM5['TZLM'][t_gem]

            missing_GEM_value_t = ds_GEM5['TCCM'].getncattr('_FillValue')
            missing_GEM_value_h = ds_GEM5['TZHM'].getncattr('_FillValue')
            missing_GEM_value_m = ds_GEM5['TZMM'].getncattr('_FillValue')
            missing_GEM_value_l = ds_GEM5['TZLM'].getncattr('_FillValue')

            data['gem']['t']    = data['gem']['t'] * calipso_mask_t * calipso_mask_82deg
            data['gem']['h']    = data['gem']['h'] * calipso_mask_h * calipso_mask_82deg
            data['gem']['m']    = data['gem']['m'] * calipso_mask_m * calipso_mask_82deg
            data['gem']['l']    = data['gem']['l'] * calipso_mask_l * calipso_mask_82deg

            G_mask_t   = np.where(data['gem']['t'] == missing_GEM_value_t, 0 , 1)
            G_mask_h   = np.where(data['gem']['h'] == missing_GEM_value_h, 0 , 1)
            G_mask_m   = np.where(data['gem']['m'] == missing_GEM_value_m, 0 , 1)
            G_mask_l   = np.where(data['gem']['l'] == missing_GEM_value_l, 0 , 1)

            data['gem']['t'] = data['gem']['t']* G_mask_t
            data['gem']['h'] = data['gem']['h']* G_mask_h
            data['gem']['m'] = data['gem']['m']* G_mask_m
            data['gem']['l'] = data['gem']['l']* G_mask_l

            ########################################################
            ######                   COSP2                    ######
            ########################################################
            data['cosp']['t'] = np.array(ds_COSP['cltcalipso'][:].T/100) * calipso_mask_t * calipso_mask_82deg
            data['cosp']['h'] = np.array(ds_COSP['clhcalipso'][:].T/100) * calipso_mask_h * calipso_mask_82deg
            data['cosp']['m'] = np.array(ds_COSP['clmcalipso'][:].T/100) * calipso_mask_m * calipso_mask_82deg
            data['cosp']['l'] = np.array(ds_COSP['cllcalipso'][:].T/100) * calipso_mask_l * calipso_mask_82deg

            missing_value_COSP = -1.000000062271131e+28

            COSP_mask_t  = np.where(data['cosp']['t'] == missing_value_COSP, 0, 1)
            COSP_mask_h  = np.where(data['cosp']['h'] == missing_value_COSP, 0, 1)
            COSP_mask_m  = np.where(data['cosp']['m'] == missing_value_COSP, 0, 1)
            COSP_mask_l  = np.where(data['cosp']['l'] == missing_value_COSP, 0, 1)

            data['cosp']['t'] = data['cosp']['t'] * COSP_mask_t 
            data['cosp']['h'] = data['cosp']['h'] * COSP_mask_h
            data['cosp']['m'] = data['cosp']['m'] * COSP_mask_m
            data['cosp']['l'] = data['cosp']['l'] * COSP_mask_l

            common_mask['t'] = calipso_mask_t * G_mask_t * calipso_mask_82deg
            common_mask['h'] = calipso_mask_h * G_mask_h * calipso_mask_82deg
            common_mask['m'] = calipso_mask_m * G_mask_m * calipso_mask_82deg
            common_mask['l'] = calipso_mask_l * G_mask_l * calipso_mask_82deg

            for layer in layers:
                n[layer] = n[layer] + common_mask[layer]
                for dataset in datasets:
                    sum_data[dataset][layer] = sum_data[dataset][layer] + data[dataset][layer]
    
    return n, sum_data, data

def create_output_file(YYYY, MM, diro, n, sum_data, data):
    datasets = ['calipso','gem','cosp']
    layers   = ['t','h','m','l']
    YYYY = YYYY
    MM   = str(MM).zfill(2)
    #################################################################################################################
    ###                                              CREATING OUTPUT (ncfile)                                     ###
    #################################################################################################################
    # nc file in to configure output file
    nci = config['GEM5']['step0']
    nc = netCDF4.Dataset(nci, 'r')
    ncfileo = diro + '/{}{}.nc'.format(YYYY,MM)
    print(diro)
    nco     = netCDF4.Dataset(ncfileo , 'w')
    # Copying global attributes
    nco.setncatts(nc.__dict__)

    # Creating dimensions
    dimnames = [ dim for dim in nc.dimensions ]
    for dimname in dimnames:
        #print(dimname)
        dimension = nc.dimensions[dimname]
        nco.createDimension(dimname, (len(dimension) if not dimension.isunlimited() else None))

    # Creating 'old' variables
    varnames = [ var for var in nc.variables ]
    for varname in varnames:
        if varname in [ 'lon', 'lat', 'rlon', 'rlat', 'rotated_pole' ]:
            variable = nc[varname]
            x = nco.createVariable(varname, variable.datatype, variable.dimensions, zlib=True, complevel=4)
            nco[varname].setncatts(variable.__dict__)


    # Creating 'new' variables
    variable = nc['lon']
    layer_names = ['Tot_cloud_cover','High_cloud_cover','Mid_cloud_cover','Low_cloud_cover']
    for layer in layers:
        for dataset in datasets:
            varname = dataset +'_'+layer_names[layers.index(layer)] + '_' + layer
            x = nco.createVariable(varname , variable.datatype, variable.dimensions, zlib=True, complevel=4)

    for layer in layers:
        varname = 'n_' + layer
        x = nco.createVariable(varname , variable.datatype, variable.dimensions, zlib=True, complevel=4)

    # Copying values for the 'old' variables
    varnames = [ var for var in nc.variables ]
    for varname in varnames:
        if varname in [ 'lon', 'lat', 'rlon', 'rlat', 'rotated_pole' ]:
            #print(varname, nc[varname][:])
            nco[varname][:] = nc[varname][:]

    # Filling values for the 'new' variables
    layer_names = ['Tot_cloud_cover','High_cloud_cover','Mid_cloud_cover','Low_cloud_cover']
    for dataset in datasets:
            for layer in layers:
                varname = dataset +'_'+layer_names[layers.index(layer)]  +'_' + layer
                nco[varname][:] = sum_data[dataset][layer]

    for layer in layers:
        varname = 'n_' + layer
        nco[varname][:] =  n[layer]
        print(np.nanmin(n[layer]))
        print(np.nanmax(n[layer]))

    nco.close()
#################################################################################################################
### MAIN ########################################################################################################
#################################################################################################################


#########################################################################
# Input arguments                                                       #
#########################################################################
working_directory =     sys.argv[1]
YYYYMM            =     sys.argv[2]
layerdef          =     sys.argv[3]
overwrite         =     sys.argv[4].lower()

YYYY = int(str(YYYYMM[0:4]))
MM   = int(str(YYYYMM[4:6]))

#########################################################################
# Configuration file (yml)                                              #
#########################################################################
yml_file = working_directory + '/../config.yml'
stream = open(yml_file,'r')
config = yaml.safe_load(stream)

domain  = config['domain' ]
dirout  = config['CALIPSOvsCOSP2vsGEM5_maps']['NetCDF']+'/'+ layerdef
dirlist = config['CALIPSO']['list']   + '/' + domain

pm               = "pm2013010100"
gemname          = "COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes"

# Initializing filepath to modify in the loop
filepath_GEM     = config['GEM5']['NetCDF'] +'/'+ '{}_YYYYMM/'.format(gemname)+'{}_YYYYMMDDd.nc'.format(pm)
filepath_CALIPSO = config['CALIPSO_CHINOOK']['NetCDF']+ '_LowMidHigh/'+ layerdef +'/YYYY'
filepath_COSP    = config['COSP2']['output'] +'/'+ domain + '/M01SC002/CALIPSO/calipso_cloudmap/YYYYMM/cospout_YYYYMMDDHH00_2D.nc'

if not os.path.exists(dirout): os.makedirs(dirout)

#Create a dataframe containing YYYYMM files
df_CALIPSO = create_dataframe(dirlist, YYYYMM)

n, sum_data, data = monthly_mean(df_CALIPSO,filepath_CALIPSO,filepath_GEM,filepath_COSP)

# Output data 
create_output_file(YYYY,MM, dirout, n, sum_data, data)
