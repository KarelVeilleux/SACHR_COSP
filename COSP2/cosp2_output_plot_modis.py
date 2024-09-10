import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd
import sys
import os
import netCDF4

from cosp2_figure_module import generate_domain_coord
from cosp2_figure_module import extract_cloudsat_track 
from cosp2_figure_module import generate_track_indices
from cosp2_figure_module import construct_profil_cloudsat
from cosp2_figure_module import construct_profil_2Ddata
from cosp2_figure_module import construct_model_layer
from cosp2_figure_module import compute_overlap_coeff
from cosp2_figure_module import format_levels
from cosp2_figure_module import plot_profil


from cosp2_figure_module import read_data2D
from cosp2_figure_module import resize_data
from cosp2_figure_module import plot_map

pm          = 'pm2013010100_'
gemname     = 'COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
gempath     = 'Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
dircosp     = '/pampa/poitras/DATA/COSP2/'              + gempath
dirgem      = '/pampa/poitras/DATA/GEM5/COSP2/' + gempath + '/Samples_NetCDF'

ncmodis = '/pampa/poitras/DATA/TREATED/MCD06COSP_D3_MODIS/COSP_DOMAIN/MCD06COSP_D3_MODIS_201401.nc'


MODIS_varname   = 'Cloud_Fraction'
filelist        = sys.argv[1]      
df              = pd.read_csv(filelist,delimiter='\s+', header=None)
df.columns      = ['ncfile_MODIS','date', 'month', 'date_gem', 't_gem']
#df             = df[   df['date_gem'] >= YYYYMMDDi].reset_index(drop=True)
#df             = df[   df['date_gem'] <= YYYYMMDDf].reset_index(drop=True)

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

GEM_nX    = np.zeros(MODIS_dim)
GEM_sumX  = np.zeros(MODIS_dim)
GEM_avgX  = np.zeros(MODIS_dim)


modis_n    = np.zeros(MODIS_dim)
modis_sum  = np.zeros(MODIS_dim)
modis_avg  = np.zeros(MODIS_dim)
print(MODIS_miss)

modis_nc   = netCDF4.Dataset(ncmodis,'r');
modis_miss = getattr(modis_nc['Cloud_Mask_Fraction_Sum'], 'missing_value')
for day in range(31):
    s  = np.array(modis_nc['Cloud_Mask_Fraction_Sum'][day])
    n  = np.array(modis_nc['Cloud_Mask_Fraction_Pixel_Counts'][day])
    modis_mask = np.where(np.array(s) == modis_miss, 0 , 1)
    modis_sum = modis_sum +s*modis_mask
    modis_n   = modis_n   +n*modis_mask
    print('ssss',modis_sum)
modis_sum = modis_sum.T
modis_n   = modis_n.T



print(MODIS_var)
for i in range(len(df['date_gem'])):
    #print(df.loc[[i]])
    # MODIS
    MODIS_nc   = netCDF4.Dataset(df['ncfile_MODIS'][i],'r');
    MODIS_var  = MODIS_nc[MODIS_varname]
    MODIS_mask = np.where(np.array(MODIS_var == MODIS_miss), 0 , 1)
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


    common_mask = MODIS_mask * GEM_mask
    GEM_sum   = GEM_sum   + common_mask * GEM_var
    MODIS_sum = MODIS_sum + common_mask * MODIS_var
    GEM_n     = GEM_n     + common_mask
    MODIS_n   = MODIS_n   + common_mask
    

plt.figure(1)
plt.imshow(MODIS_sum/MODIS_n)
plt.gca().invert_yaxis()


plt.figure(2)
plt.imshow(GEM_sum/GEM_n)
plt.gca().invert_yaxis()

plt.figure(3)
plt.imshow(modis_sum/modis_n)
plt.gca().invert_yaxis()



plt.show()
exit()
grid_gem        = 'NAM-11m'
grid_cosp       = 'box11'
gempath         = 'Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
ncfile_cloudsat = '/pampa/poitras/DATA/ORIGINAL/CLOUDSAT/NetCDF/CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00/20140105180611_40917_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.nc'
ncfile_cospout  = '/pampa/poitras/DATA/TREATED/COSP2/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/MPB/OUTPUT/cosp_output_201401051900_cloudsat_2D.nc'
ncfile_cospin   = '/pampa/poitras/DATA/TREATED/COSP2/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/MPB/INPUT/cosp_input_201401051900.nc'
ncfile_gem      = '/pampa/poitras/DATA/TREATED/GEM5/' + gempath + '/Samples_NetCDF_COSP/storm_019_201401/pm2013010100_20140105d.nc'
t_gem = 19-1


domain_coord        = generate_domain_coord(grid_cosp)
cloudsat_track      = extract_cloudsat_track(ncfile_cloudsat, domain_coord)
cloudsat_track_indx = generate_track_indices(cloudsat_track,  grid_cosp)



##############################


MP='MPB'
MSC='1M50SC'
grid_cosp       = 'box11'
gempath         = 'Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
ncfile_gem      = '/pampa/poitras/DATA/TREATED/GEM5/' + gempath + '/Samples_NetCDF_COSP/storm_019_201401/pm2013010100_20140105d.nc'
ncfile_modis    = '/pampa/poitras/DATA/TREATED/MYD06/MYD06_L2_201401051900_201401051910.nc'
ncfile_mcd06    = '/pampa/poitras/DATA/TREATED/MCD06COSP_D3_MODIS/COSP_DOMAIN/MCD06COSP_D3_MODIS_201401.nc'
ncfile_cospout50 = '/pampa/poitras/DATA/TREATED/COSP2/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/' + MP + '/OUTPUT/' + MSC + '/cosp_output_201401051900_modis_2D.nc'
dirfig           = '/pampa/poitras/figures/COSP2/' +  gempath   + '/' + MP + '/OUTPUT/' + MSC
date             = '201401051900'
t_gem = 19-1
t_mcd06 = 5-1






vars_gem     = {'cloud_fraction_total': 'NFRT'}
vars_cospout = {'cloud_fraction_total': 'cltmodis'                }                    
vars_modis   = {'cloud_fraction_total': 'Cloud_Fraction'          }      
vars_mcd06   = {'cloud_fraction_total': 'Cloud_Mask_Fraction_Mean'}      
titles       = {'cloud_fraction_total': 'Total cloud fraction'    }


#vars_gem     = {'cloud_fraction_low': 'NFRL'}
#vars_cospout = {'cloud_fraction_low': 'cllmodis'                }
#vars_modis   = {'cloud_fraction_low': 'Cloud_Fraction'          }
#vars_mcd06   = {'cloud_fraction_low': 'Cloud_Mask_Fraction_Low_Mean'}
#titles       = {'cloud_fraction_low': 'Low levels cloud fraction'    }


#vars_gem     = {'cloud_fraction_mid': 'NFRM'}
#vars_cospout = {'cloud_fraction_mid': 'clmmodis'                }
#vars_modis   = {'cloud_fraction_low': 'Cloud_Fraction'          }
#vars_mcd06   = {'cloud_fraction_mid': 'Cloud_Mask_Fraction_Mid_Mean'}
#titles       = {'cloud_fraction_mid': 'Mid levels cloud fraction'    }


#vars_gem     = {'cloud_fraction_high': 'NFRH'}
#vars_cospout = {'cloud_fraction_high': 'clhmodis'                }
#vars_modis   = {'cloud_fraction_low': 'Cloud_Fraction'          }
#vars_mcd06   = {'cloud_fraction_high': 'Cloud_Mask_Fraction_High_Mean'}
#titles       = {'cloud_fraction_high': 'High levels cloud fraction'    }


#vars_cospout = {'cloud_water_path_liquid': 'lwpmodis'}
#vars_mcd06   = {'cloud_water_path_liquid': 'Cloud_Water_Path_Liquid_Mean'}
#titles       = {'cloud_water_path_liquid': 'Liquid cloud water path [kg/m2]'}

#vars_cospout = {'cloud_water_path_ice': 'iwpmodis'}
#vars_mcd06   = {'cloud_water_path_ice': 'Cloud_Water_Path_Ice_Mean'}
#titles       = {'cloud_water_path_ice': 'Ice cloud water path [kg/m2]'}


vars_cospout = {'cloud_optical_thickness_total':'tautmodis'}
vars_modis   = {'cloud_optical_thickness_total':'Cloud_Optical_Thickness'}
vars_mcd06   = {'cloud_optical_thickness_total':'Cloud_Optical_Thickness_Total_Mean'}
titles       = {'cloud_optical_thickness_total':'Total cloud optical thickness'}


vars_cospout = {'cloud_optical_thickness_liquid':'tauwmodis'}
vars_mcd06   = {'cloud_optical_thickness_liquid':'Cloud_Optical_Thickness_Liquid_Mean'}
titles       = {'cloud_optical_thickness_liquid':'Liquid cloud optical thickness'}

vars_cospout = {'cloud_optical_thickness_ice':'tauimodis'}
vars_mcd06   = {'cloud_optical_thickness_ice':'Cloud_Optical_Thickness_Ice_Mean'}
titles       = {'cloud_optical_thickness_ice':'Ice cloud optical thickness'}


vars_cospout = {'cloud_particle_size_ice':'reffclimodis'}
vars_mcd06   = {'cloud_particle_size_ice':'Cloud_Particle_Size_Ice_Mean'}
titles       = {'cloud_particle_size_ice':'Ice cloud particle size [micron]'}

vars_cospout = {'cloud_particle_size_liquid':'reffclwmodis'}
vars_mcd06   = {'cloud_particle_size_liquid':'Cloud_Particle_Size_Liquid_Mean'}
titles       = {'cloud_particle_size_liquid':'Liquid cloud particle size [micron]'}

#vars_gem     = {'cloud_fraction_total': 'NFRT'}
#vars_cospout = {'cloud_fraction_total': 'cltmodis'            , 'cloud_water_path_total': 'lwpmodis_plus_iwpmodis', 'cloud_optical_thickness_total':'tautmodis'}
#vars_modis   = {'cloud_fraction_total': 'Cloud_Fraction'      , 'cloud_water_path_total': 'Cloud_Water_Path'      , 'cloud_optical_thickness_total':'Cloud_Optical_Thickness'}
#vars_mcd06   = {'cloud_fraction_total': 'Cloud_Mask_Fraction_Mean' , 'cloud_water_path_liquid': 'Cloud_Water_Liquid_Mean' , 'cloud_optical_thickness_total':'Cloud_Optical_Thickness_Mean'}
#titles       = {'cloud_fraction_total': 'Total cloud fraction', 'cloud_water_path_total': 'Total cloud water path [kg/m2]', 'cloud_optical_thickness_total':'Total cloud optical thickness'}


vars_cospout  = {'cloud_top_pressure': 'pctmodis' }
#vars_modis     = {'cloud_top_pressure': 'Cloud_Top_Pressure' }
vars_mcd06    =  {'cloud_top_pressure': 'Cloud_Top_Pressure_Mean'} 
titles        = {'cloud_top_pressure': 'Cloud top pressure [hPa]' }




for var in vars_mcd06:
    vgem     = vars_gem    [var] if var in vars_gem     else 'dummy'
    vcospout = vars_cospout[var] if var in vars_cospout else 'dummy'
    vmodis   = vars_modis  [var] if var in vars_modis   else 'dummy'
    vmcd06   = vars_mcd06  [var] if var in vars_mcd06   else 'dummy'
    title    = titles      [var] if var in titles       else ''

    modis_data      = {}
    modis_attribute = {}
    modis_data[var], modis_attribute[var] = read_data2D(ncfile_modis,'modis',vmodis)
    print('myd06',modis_data[var].shape)
    if not vmodis == 'dummy': modis_data = resize_data(modis_data,grid_cosp)

    mcd06_data      = {}
    mcd06_attribute = {}
    mcd06_data[var], modis_attribute[var] = read_data2D(ncfile_mcd06,'mcd06',vmcd06,t_mcd06)
    print('mcd06',mcd06_data[var].shape)
    if not vmcd06 == 'dummy': mcd06_data = resize_data(mcd06_data,grid_cosp)
   


    gem_data      = {}
    gem_attribute = {}
    gem_data[var]    , gem_attribute[var]     = read_data2D(ncfile_gem,'gem',vgem,t_gem)
    #gem_data[var]    , gem_attribute[var]     = read_data2D(ncfile_gem,'gem','NFRT',t_gem)
    #gem_data[var+'a'], gem_attribute[var+'a'] = read_data2D(ncfile_gem,'gem','NFR' ,t_gem)
    if not vgem == 'dummy': gem_data = resize_data(gem_data,grid_cosp)



    cospout50_data      = {}
    cospout50_attribute = {}
    cospout50_data[var], cospout50_attribute[var] = read_data2D(ncfile_cospout50,'cospout', vcospout)


    fig_att_modis  = {'title': title + '\n MODIS [MYD06_L2]', 'figname': dirfig + '/map/' + var + '_' + date + '_MYD06' }
    fig_att_mcd06  = {'title': title + '\n MODIS [MCD06_L3]', 'figname': dirfig + '/map/' + var + '_' + date + '_MCD06' }
    fig_att_cospout= {'title': title + '\n COSP2'           , 'figname': dirfig + '/map/' + var + '_' + date + '_COSP2' }
    fig_att_gem    = {'title': title + '\n GEM5'            , 'figname': dirfig + '/map/' + var + '_' + date + '_GEM5'  }
    #fig_att_gem    = {'title': 'Total cloud cover\n GEM5 (NFRT)'     , 'figname': dirfig + '/map/total_cloud_cover_' + date + '_GEM5_NFRT'}           
    #fig_att_gema   = {'title': 'Total cloud cover\n GEM5 (NFR) '     , 'figname': dirfig + '/map/total_cloud_cover_' + date + '_GEM5_NFR' }
    
    print

    plot_map(modis_data    [var]    , grid_cosp, var    , cloudsat_track_indx, fig_att_modis  )
    plot_map(mcd06_data    [var]    , grid_cosp, var    , cloudsat_track_indx, fig_att_mcd06  )
    plot_map(cospout50_data[var]    , grid_cosp, var    , cloudsat_track_indx, fig_att_cospout)
    plot_map(gem_data      [var]    , grid_cosp, var    , cloudsat_track_indx, fig_att_gem    )
    #plot_map(gem_data      [var+'a'], grid_cosp, var+'a', cloudsat_track_indx, fig_att_gema   )


    # Diff

    if (not vcospout == 'dummy') and (not vmodis == 'dummy'): 
        fig_att_cospout= {'title': title + '\n COSP2 - MODIS'           , 'figname': dirfig + '/map/diff_'         + var + '_' + date + '_COSP2_MYD06'    , 'datatype': 'difference'}
        plot_map(cospout50_data[var]     - modis_data[var], grid_cosp, var    , cloudsat_track_indx, fig_att_cospout)

    if (not vgem     == 'dummy') and (not vmodis == 'dummy'): 
        fig_att_gem    = {'title': title + '\n GEM5  - MODIS'     , 'figname': dirfig + '/map/diff_total_'   + var + '_' + date + '_GEM5_MYD06', 'datatype': 'difference'}
        plot_map(gem_data      [var]     - modis_data[var], grid_cosp, var    , cloudsat_track_indx, fig_att_gem    )

    if (not vcospout == 'dummy') and (not vmcd06 == 'dummy'):
        fig_att_cospout= {'title': title + '\n COSP2 - MCD06'           , 'figname': dirfig + '/map/diff_'         + var + '_' + date + '_COSP2_MCD06'    , 'datatype': 'difference'}
        plot_map(cospout50_data[var]     - mcd06_data[var], grid_cosp, var    , cloudsat_track_indx, fig_att_cospout)


    if (not vgem     == 'dummy') and (not vmcd06 == 'dummy'):
        fig_att_gem    = {'title': title + '\n GEM5 - MCD06'     , 'figname': dirfig + '/map/diff_total_'   + var + '_' + date + '_GEM5_MCD06', 'datatype': 'difference'}
        plot_map(gem_data      [var]     - mcd06_data[var], grid_cosp, var    , cloudsat_track_indx, fig_att_gem    )



    #if (not vgem == 'dummy') and (not vcospout == 'dummy') and (not vmodis == 'dummy'):
    #    fig_att_gem    = {'title': title + '\n |COSP2 - MODIS| - |GEM5 - MODIS|'     , 'figname': dirfig + '/map/diff_' + var + '_' + date + '_MYD06_COSP2_MYD06_GEM5', 'datatype': 'difference'}
    #    plot_map(np.abs(modis_data[var] - cospout50_data[var])  - np.abs(modis_data[var] - gem_data      [var])  , grid_cosp, var    , cloudsat_track_indx, fig_att_gem )







plt.show()

exit()




#######################################################################################################################################################################
domain_coord   = generate_domain_coord(grid_cosp)
cloudsat_track = extract_cloudsat_track(ncfile_cloudsat, domain_coord)
indx_cospin    = generate_track_indices(cloudsat_track,  grid_cosp) 
indx_gem       = generate_track_indices(cloudsat_track,  grid_gem )


indx_cospout = {'i': indx_cospin['j'], 'j': indx_cospin['i']}


#######################################################################################################################################################################
#                                                               Radar reflectivity                                                                                    #
#######################################################################################################################################################################
profil_cloudsat = {}
profil_cloudsat['orography'         ] = construct_profil_cloudsat(ncfile_cloudsat, cloudsat_track, 'DEM_elevation'     )
profil_cloudsat['radar_reflectivity'] = construct_profil_cloudsat(ncfile_cloudsat, cloudsat_track, 'Radar_Reflectivity') 

flag_nan = np.isnan(profil_cloudsat['radar_reflectivity'] )
profil_cloudsat['radar_reflectivity'][flag_nan] = -1000 


profil_cospout = {}
profil_cospout['orography'         ] = construct_profil_2Ddata(ncfile_cospin , indx_cospin , 'orography')
profil_cospout['radar_reflectivity'] = construct_profil_2Ddata(ncfile_cospout, indx_cospout, 'dbze94')

profil_gem = {}
profil_gem['orography'         ]     = profil_cospout['orography']
profil_gem['radar_reflectivity']     = construct_profil_2Ddata(ncfile_gem, indx_gem, 'ZET', timeslice_gem)



cloudsat_layer = np.arange(0,(105+1)*240,240)
model_layer    = construct_model_layer(ncfile_cospin, indx_cospin)
overlap_coeff  = compute_overlap_coeff(np.flipud(model_layer), np.flip(cloudsat_layer))


PROFIL_cospout = {}
PROFIL_gem     = {}
PROFIL_cospout['radar_reflectivity'] = format_levels(profil_cospout['radar_reflectivity'], overlap_coeff)
PROFIL_gem    ['radar_reflectivity'] = format_levels(profil_gem    ['radar_reflectivity'], overlap_coeff)
PROFIL_cospout['orography']          = profil_cospout['orography']
PROFIL_gem    ['orography']          = profil_gem    ['orography']


image_attribute_cloudsat = {'title':'CloudSat: Radar reflectivity', 'figurename':'...'}
image_attribute_cosp     = {'title':'COSP: Radar reflectivity '   , 'figurename':'...'}
image_attribute_gem      = {'title':'GEM: Radar reflectivity '    , 'figurename':'...'}

plot_profil(profil_cloudsat,'radar_reflectivity', image_attribute_cloudsat)
plot_profil(PROFIL_cospout ,'radar_reflectivity', image_attribute_cosp    )
plot_profil(PROFIL_gem     ,'radar_reflectivity', image_attribute_gem     )
plt.show()




#for i in range(len(cloudsat_track['index'])):
#    print(i,cloudsat_track['index'][i], cloudsat_track['longitude'][i], cloudsat_track['latitude'][i], cloudsat_track['time'][i])
