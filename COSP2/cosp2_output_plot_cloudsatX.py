import matplotlib.pyplot as     plt
import numpy             as     np
from cosp2_figure_module import generate_domain_coord
from cosp2_figure_module import extract_cloudsat_track 
from cosp2_figure_module import generate_track_indices
from cosp2_figure_module import construct_profil_cloudsat
from cosp2_figure_module import construct_profil_2Ddata
from cosp2_figure_module import construct_model_layer
from cosp2_figure_module import compute_overlap_coeff
from cosp2_figure_module import format_levels
from cosp2_figure_module import plot_profil



MP='MPB'
MSC='1M001SC'
date            = '201401051900'
grid_gem        ='NAM-11m'
grid_cosp       ='box11'
gempath         ='Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
dircosp         ='/pampa/poitras/DATA/TREATED/COSP2/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
ncfile_cloudsat ='/pampa/poitras/DATA/ORIGINAL/CLOUDSAT/NetCDF/CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00/20140105180611_40917_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.nc'
ncfile_cospout  = dircosp + '/' + MP + '/OUTPUT/' + MSC + '/cosp_output_201401051900_cloudsat_2D.nc'
ncfile_cospin   = dircosp + '/' + MP + '/INPUT/cosp_input_201401051900.nc'
ncfile_gem      ='/pampa/poitras/DATA/TREATED/GEM5/' + gempath + '/Samples_NetCDF_COSP/storm_019_201401/pm2013010100_20140105d.nc'
dirfig          ='/pampa/poitras/figures/COSP2/' +  gempath   + '/' + MP + '/OUTPUT/' + MSC

timeslice_gem = 19-1
dircosp='/pampa/poitras/DATA/TREATED/COSP2/Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
ncfile_cospout  = dircosp + '/' + MP + '/OUTPUT/' + MSC + '/' 'cosp_output_201401051900_cs_test2_2D.nc'
ncfile_cospin   = '/pampa/poitras/SCRATCH/COSP2/cospin_test.nc'


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
PROFIL_cospout['radar_reflectivity'] = format_levels(profil_cospout['radar_reflectivity'], overlap_coeff,105)
PROFIL_gem    ['radar_reflectivity'] = format_levels(profil_gem    ['radar_reflectivity'], overlap_coeff,105)
PROFIL_cospout['orography']          = profil_cospout['orography']
PROFIL_gem    ['orography']          = profil_gem    ['orography']
#

image_attribute_cloudsat = {'title':'CloudSat: Radar reflectivity', 'fignamex': dirfig + '/profile/' + 'radar_reflectivity' + '_' + date + '_cloudsat_test' }
image_attribute_cosp     = {'title':'COSP: Radar reflectivity '   , 'fignamex': dirfig + '/profile/' + 'radar_reflectivity' + '_' + date + '_COSP2' }
image_attribute_gem      = {'title':'GEM: Radar reflectivity '    , 'fignamex': dirfig + '/profile/' + 'radar_reflectivity' + '_' + date + '_GEM5' }

plot_profil(profil_cloudsat,'radar_reflectivity', image_attribute_cloudsat)
plot_profil(PROFIL_cospout ,'radar_reflectivity', image_attribute_cosp    )
plot_profil(PROFIL_gem     ,'radar_reflectivity', image_attribute_gem     )
plt.show()




#for i in range(len(cloudsat_track['index'])):
#    print(i,cloudsat_track['index'][i], cloudsat_track['longitude'][i], cloudsat_track['latitude'][i], cloudsat_track['time'][i])
