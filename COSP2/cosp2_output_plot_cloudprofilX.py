import matplotlib.pyplot as     plt
import numpy             as     np
from cosp2_figure_module import generate_domain_coord
from cosp2_figure_module import extract_cloudsat_track
from cosp2_figure_module import extract_satellite_track
from cosp2_figure_module import generate_track_indices
from cosp2_figure_module import construct_profil_cloudsat
from cosp2_figure_module import construct_profil_satellite
from cosp2_figure_module import construct_profil_2Ddata
from cosp2_figure_module import construct_model_layer
from cosp2_figure_module import compute_overlap_coeff
from cosp2_figure_module import format_levels
from cosp2_figure_module import plot_profil
from cosp2_figure_module import plot_frequency_intensity_profil
from cosp2_figure_module import mask_orography

MP='MPB'
MSC='1M002SC'
date            = '201401051900'
grid_gem        ='NAM-11m'
grid_cosp       ='box11'
gempath         ='Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
dircosp         ='/pampa/poitras/DATA/TREATED/COSP2/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
ncfile_cloudsat ='/pampa/poitras/DATA/ORIGINAL/CLOUDSAT/NetCDF/CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00/20140105180611_40917_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.nc'
ncfile_calipso  ='/pampa/poitras/DATA/ORIGINAL/CAL_LID_L2_05kmCPro-Standard-V4/NetCDF/CAL_LID_L2_05kmCPro-Standard-V4-20.2014-01-05T18-21-52ZD.nc'
ncfile_cospout  = dircosp + '/' + MP + '/OUTPUT/' + MSC + '/cosp_output_201401051900_cloudprofile_2D.nc'
ncfile_cospin   = dircosp + '/' + MP + '/INPUT/cosp_input_201401051900.nc'
ncfile_gem      ='/pampa/poitras/DATA/TREATED/GEM5/' + gempath + '/Samples_NetCDF_COSP/storm_019_201401/pm2013010100_20140105d.nc'
dirfig          ='/pampa/poitras/figures/COSP2/' +  gempath   + '/' + MP + '/OUTPUT/' + MSC


timeslice_gem = 19-1


#######################################################################################################################################################################
domain_coord   = generate_domain_coord(grid_cosp)


#cloudsat_track = extract_cloudsat_track(ncfile_cloudsat, domain_coord)
#indx_cospin    = generate_track_indices(cloudsat_track,  grid_cosp)
#indx_gem       = generate_track_indices(cloudsat_track,  grid_gem )
#indx_cospout = {'i': indx_cospin['j'], 'j': indx_cospin['i']}


calipso_track  = extract_satellite_track(ncfile_calipso, domain_coord,'calipso')
indx_cospin    = generate_track_indices(calipso_track,  grid_cosp) 
indx_gem       = generate_track_indices(calipso_track,  grid_gem )
indx_cospout = {'i': indx_cospin['j'], 'j': indx_cospin['i']}


#######################################################################################################################################################################
#                                                               Cloud cover (profile)                                                                                 #
#######################################################################################################################################################################

# READING INPUT DATA
profil_calipso = {}
profil_calipso['orography'         ] = construct_profil_satellite(ncfile_calipso, calipso_track, 'Surface_Elevation_Statistics','calipso')
profil_calipso['total_cloud_cover' ] = construct_profil_satellite(ncfile_calipso, calipso_track, 'Cloud_Layer_Fraction',        'calipso') 

profil_cospout = {}
profil_cospout['orography'        ] = construct_profil_2Ddata(ncfile_cospin , indx_cospin , 'orography')
profil_cospout['total_cloud_cover'] = construct_profil_2Ddata(ncfile_cospout, indx_cospout, 'clcalipso')
profil_cospout['total_cloud_cover' ][profil_cospout['total_cloud_cover' ]<0] = 0

profil_gem = {}
profil_gem['orography'        ] = profil_cospout['orography']
profil_gem['total_cloud_cover'] = construct_profil_2Ddata(ncfile_gem, indx_gem, 'FN', timeslice_gem)




# INTERPOLATION OF VERTICAL LAYERS
layer_gem     = construct_model_layer(ncfile_cospin, indx_cospin)
layer_cospout = np.arange(   0,(320+1)*60,60)
layer_calipso = np.arange(-480,-480+(398+1)*60,60)
layer_target  = np.arange(   0,(320+1)*60,60)

overlap_coeff_cospout = compute_overlap_coeff(np.flipud(layer_cospout), np.flip(layer_target))
overlap_coeff_calipso = compute_overlap_coeff(np.flipud(layer_calipso), np.flip(layer_target))
overlap_coeff_gem     = compute_overlap_coeff(np.flipud(layer_gem    ), np.flip(layer_target))

PROFIL_gem     = {}
PROFIL_gem['total_cloud_cover']     = format_levels(profil_gem['total_cloud_cover']    , overlap_coeff_gem    ,320)
PROFIL_gem['orography'        ]     = profil_gem['orography']
mask_orography(PROFIL_gem,layer_target,'total_cloud_cover')


PROFIL_cospout = {}
PROFIL_cospout['total_cloud_cover'] = format_levels(profil_cospout['total_cloud_cover'], overlap_coeff_cospout,320)
PROFIL_cospout['orography'        ] = profil_cospout['orography']
mask_orography(PROFIL_cospout,layer_target,'total_cloud_cover')


PROFIL_calipso = {}
PROFIL_calipso['total_cloud_cover'] = format_levels(profil_calipso['total_cloud_cover'], overlap_coeff_calipso,320)
PROFIL_calipso['orography'        ] = profil_calipso['orography']
mask_orography(PROFIL_calipso,layer_target,'total_cloud_cover')




# INTENSITY-FREQUENCY FIGURES

#for threshold in np.arange(0,1.05,0.05):
threshold = 1e-6
th       = str(threshold).replace('.','p')
data    = {'calipso':  PROFIL_calipso, 'cospout':  PROFIL_cospout}
data    = {'calipso':  PROFIL_calipso, 'cospout':  PROFIL_cospout, 'gem':  PROFIL_gem}
img_att = {'title':'Total cloud cover ' , 'vstruct':'calipso', 'figname': dirfig + '/freq_int/' + 'cloud_cover_total_xxxYYY_' + th + '_' + date + '_calipso_COSP2_GEM5'   }
plot_frequency_intensity_profil(data, 'total_cloud_cover', threshold, img_att)

#plt.show()
#exit()



image_attribute_calipso  = {'title':'CALIPSO: Total cloud cover'           , 'vstruct':'calipso', 'figname': dirfig + '/profile/' + 'cloud_cover' + '_' + date + '_calipso'}
image_attribute_cosp     = {'title':'COSP: Total cloud cover (CALIPSO) '   , 'vstruct':'calipso', 'figname': dirfig + '/profile/' + 'cloud_cover' + '_' + date + '_COSP2'  }
image_attribute_gem      = {'title':'GEM: Total cloud cover '              , 'vstruct':'calipso', 'figname': dirfig + '/profile/' + 'cloud_cover' + '_' + date + '_GEM5'   }

plot_profil(PROFIL_calipso,'total_cloud_cover', image_attribute_calipso)
plot_profil(PROFIL_cospout,'total_cloud_cover', image_attribute_cosp)
plot_profil(PROFIL_gem    ,'total_cloud_cover', image_attribute_gem )

#plt.figure(4)
#plt.plot(profil_calipso['orography'],'r-')
#plt.plot(profil_cospout['orography'        ],'b-')
plt.show()





