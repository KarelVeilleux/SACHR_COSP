import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd
import sys
import os
import netCDF4




from cosp2_figure_module import plot_map

MP          ='MPB'
MSC         ='1M002SC'
pm          = 'pm2013010100_'
gemname     = 'COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
gempath     = 'Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
dircosp     = '/pampa/poitras/DATA/COSP2/'      + gempath + '/' + MP + '/' + 'OUTPUT/NAM11/' + MSC
dirgem      = '/pampa/poitras/DATA/GEM5/COSP2/' + gempath + '/Samples_NetCDF'

ncmodis = '/pampa/poitras/DATA/TREATED/MCD06COSP_D3_MODIS/COSP_DOMAIN/MCD06COSP_D3_MODIS_201401.nc'
YYYYMMDDi = 20140101
YYYYMMDDf = 20140131
MODIS_varname   = 'Cloud_Fraction'
COSP_varname    = 'cltmodis'
filelist        = sys.argv[1]      
df              = pd.read_csv(filelist,delimiter='\s+', header=None)
df.columns      = ['ncfile_MODIS','date', 'month', 'date_gem', 't_gem']
df             = df[   df['date_gem'] >= YYYYMMDDi].reset_index(drop=True)
df             = df[   df['date_gem'] <= YYYYMMDDf].reset_index(drop=True)

print(df)



# Retrieve field dimension
MODIS_nc   = netCDF4.Dataset(df['ncfile_MODIS'][0],'r')
MODIS_var  = MODIS_nc[MODIS_varname]
MODIS_dim  = MODIS_var[:].shape
MODIS_miss = getattr(MODIS_var, 'missing_value')
MODIS_n    = np.zeros(MODIS_dim)
MODIS_sum  = np.zeros(MODIS_dim)
MODIS_avg  = np.zeros(MODIS_dim)

GEM_n    = np.zeros(MODIS_dim)
GEM_sum  = np.zeros(MODIS_dim)
GEM_avg  = np.zeros(MODIS_dim)

COSP_n    = np.zeros(MODIS_dim)
COSP_sum  = np.zeros(MODIS_dim)
COSP_avg  = np.zeros(MODIS_dim)


modis_n    = np.zeros(MODIS_dim)
modis_sum  = np.zeros(MODIS_dim)
modis_avg  = np.zeros(MODIS_dim)

for i in range(len(df['date_gem'])):
    #print(df.loc[[i]])
    # MODIS
    MODIS_nc    = netCDF4.Dataset(df['ncfile_MODIS'][i],'r');
    MODIS_var   = MODIS_nc[MODIS_varname]
    MODIS_mask  = np.where(np.array(MODIS_var == MODIS_miss), 0 , 1)
    MODIS_flag  = MODIS_nc['flag_day_night'][:]
    MODIS_day   = np.where(np.array(MODIS_flag == 1), 1 , 0)
    MODIS_night = np.where(np.array(MODIS_flag == 0), 1 , 0)
    #MODIS_sum  = MODIS_sum + MODIS_var*MODIS_mask
    #MODIS_n    = MODIS_n   + MODIS_mask




    #GEM
    t_gem          =     df['t_gem'     ][i]
    YYYYMMDD_gem   = str(df['date_gem'  ][i])
    YYYYMM_gem     = str(df['date_gem'  ][i])[0:6]
    date           = str(df['date'      ][i])
    ncfile_GEM     = dirgem  + '/' + gemname + '_' + YYYYMM_gem + '/' + pm + YYYYMMDD_gem + 'd.nc'

    
    GEM_varname= 'TCCM'
    GEM_nc     = netCDF4.Dataset(ncfile_GEM,'r');
    GEM_var    = GEM_nc[GEM_varname][t_gem]
    GEM_miss   = -127
    GEM_mask   = np.where(np.array(GEM_var == GEM_miss), 0 , 1)
    #GEM_sum    = GEM_sum + GEM_var*GEM_mask
    #GEM_n      = GEM_n   + GEM_mask




    #COSP
    date        = df['date'][i] 
    ncfile_COSP = dircosp + '/cosp_output_' + str(date) + '_modis_2D.nc'
    COSP_nc     = netCDF4.Dataset(ncfile_COSP,'r'); 
    COSP_var    = COSP_nc['cltmodis'][:].T*0.01
    COSP_miss   = -1e+30*0.01
    COSP_mask   = np.where(COSP_var == COSP_miss, 0 , 1)
    #plt.imshow(COSP_var)
    #plt.figure(2)
    #plt.imshow(COSP_mask)
    #print(COSP_var == COSP_miss)

    #plt.show()

    common_mask = MODIS_mask * GEM_mask * COSP_mask *  MODIS_day
    
    
    GEM_sum   = GEM_sum   + common_mask * GEM_var
    MODIS_sum = MODIS_sum + common_mask * MODIS_var
    COSP_sum  = COSP_sum  + common_mask * COSP_var
    GEM_n     = GEM_n     + common_mask
    MODIS_n   = MODIS_n   + common_mask
    COSP_n    = COSP_n    + common_mask

    #GEM_sum   = GEM_sum   + GEM_mask * GEM_var
    #MODIS_sum = MODIS_sum + MODIS_mask * MODIS_var
    #COSP_sum  = COSP_sum  + COSP_mask * COSP_var
    #GEM_n     = GEM_n     + GEM_mask
    #MODIS_n   = MODIS_n   + MODIS_mask
    #COSP_n    = COSP_n    + COSP_mask



MODIS_average = MODIS_sum/MODIS_n
GEM_average   = GEM_sum  /GEM_n
COSP_average   = COSP_sum  /COSP_n
GEM_difference    = GEM_average - MODIS_average
COSP_difference    = COSP_average - MODIS_average
xxxx_difference = COSP_average -  GEM_average


plot_map(MODIS_average  , 'NAM11', 'cloud_fraction'     )
plot_map(GEM_average    , 'NAM11', 'cloud_fraction'     )
plot_map(COSP_average   , 'NAM11', 'cloud_fraction'     )
plot_map(GEM_difference , 'NAM11', 'cloud_fraction', attribute={'datatype':'difference'} )
plot_map(COSP_difference , 'NAM11', 'cloud_fraction', attribute={'datatype':'difference'})
plot_map(xxxx_difference , 'NAM11', 'cloud_fraction', attribute={'datatype':'difference'})
plt.show()
