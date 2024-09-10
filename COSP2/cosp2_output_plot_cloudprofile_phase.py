import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd
import os
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
from cosp2_figure_module import plot_borders_and_tracks
from cosp2_figure_module import mask_orography
from cosp2_figure_module import mask_sealand
NaN = np.nan


s='DJF'; YYYYMMDDi=20140101; YYYYMMDDf=20140132;
s='DJF'; YYYYMMDDi=20140101; YYYYMMDDf=20140102;
#s='MAM'; YYYYMMDDi=20140301; YYYYMMDDf=20140532;
#s='JJA'; YYYYMMDDi=20140601; YYYYMMDDf=20140832;
#s='SON'; YYYYMMDDi=20140901; YYYYMMDDf=20141132;
#s='ANN'; YYYYMMDDi=20140101; YYYYMMDDf=20141232;
domain='bc_coast_11'
#domain='bermuda_azores_11'
#domain='great_lakes_11'
#domain='hudson_bay_11'
#domain='pacific_sw_11'
#domain='sonora_desert_11'

input_list_file = '/pampa/poitras/DATA/CALIPSO/CAL_LID_L2_05kmCPro-Standard-V4/list/' + domain + '_2014.txt'
df              = pd.read_csv(input_list_file,delimiter='\s+', header=None)
df.columns      = ['nc_CALIPSO', 'ndata', 'ti', 'tf', 'date', 'MM', 'date_gem', 't_gem']
df             = df[   df['date_gem'] >= YYYYMMDDi].reset_index(drop=True)
df             = df[   df['date_gem'] <= YYYYMMDDf].reset_index(drop=True)
#df             = df[   df['ndata'] != 183].reset_index(drop=True)    

#print(df)

# TEMPORARY FIX
existing_files = []
for i in range(len(df['date_gem'])):
    file='/pampa/poitras/DATA/COSP2/Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/MPB/INPUT/' + 'NAM11' + '/2014/cosp_input_'  + str(df['date'][i]) + '.nc'
    if not os.path.isfile(file):
        existing_files.append(i)
df.drop(existing_files, axis=0, inplace=True)
df = df.reset_index(drop=True)





print(df)


#exit()

sum=0
for i in range(len(df['date_gem'])):
    sum = sum + df['ndata'][i]

print('npoints=',sum)


masktype    = 'show_land' # show_land, show_sea, show_sealand, show_nothing
MP          = 'MPB'
MSC         = '1M001SC'
grid_gem    = 'NAM-11m'
grid_cosp   = domain
pm          = 'pm2013010100_'
gemname     = 'COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
gempath     = 'Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
dircosp     = '/pampa/poitras/DATA/COSP2/'              + gempath
dirgem      = '/pampa/poitras/DATA/GEM5/COSP2/' + gempath + '/Samples_NetCDF'
dirfig      = '/pampa/poitras/FIGURES/COSP2/'           + gempath + '/' + MP + '/' + domain + '/OUTPUT/' + MSC


cospoutvar = 'clcalipsoliq'
GOCCPvar      = 'clcalipso_liq'
if   masktype == 'show_land'   : maskfname = 'land';
elif masktype == 'show_sea'    : maskfname = 'sea';
elif masktype == 'show_sealand': maskfname = 'sealand';
elif masktype == 'show_nothing': maskfname = 'nothing';


PROFIL_gemX={}
PROFIL_cospoutX={}
PROFIL_calipsoX={}
calipso_trackX = { 'longitude':[], 'latitude':[]}
for i in range(len(df['date_gem'])):
    print(df.loc[[i]])
    t_gem          =     df['t_gem'     ][i]
    YYYYMMDD_gem   = str(df['date_gem'  ][i])
    YYYYMM_gem     = str(df['date_gem'  ][i])[0:6]
    date           = str(df['date'      ][i])

    ncfile_calipso = df['nc_CALIPSO'][i]
    ncfile_GOCCP   = '/pampa/poitras/DATA/CALIPSO/CALIPSO_GOCCP/NAM11/3D_CloudFraction_Phase330m/avg/2014/3D_CloudFraction_Phase330m_20140101_avg_CFMIP1_sat_3.1.2.nc'
    ncfile_cospin  = dircosp + '/' + MP + '/INPUT/'  + 'NAM11' + '/2014/cosp_input_'  + date + '.nc'
    ncfile_cospout = dircosp + '/' + MP + '/OUTPUT/' + 'NAM11' + '/' + MSC + '/cosp_output_' + date + '_calipso_cloudprofile_phase_2D.nc'
    #ncfile_gem     = dirgem  + '/' + gemname + '_' + YYYYMM_gem + '/' + pm + YYYYMMDD_gem + 'd.nc'

    print(ncfile_calipso)
    print(ncfile_GOCCP)
    #print(ncfile_cospin )
    #print(ncfile_cospout)
    #print(ncfile_gem    )
    


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



    LON = np.append(calipso_trackX['longitude'], NaN); LON = np.append(LON, calipso_track['longitude']);
    LAT = np.append(calipso_trackX['latitude' ], NaN); LAT = np.append(LAT, calipso_track['latitude' ]);
    calipso_trackX = {'longitude':LON, 'latitude':LAT}
    


#######################################################################################################################################################################
#                                                               Cloud cover (profile)                                                                                 #
#######################################################################################################################################################################

# READING INPUT DATA
    #profil_calipso = {}
    #profil_calipso['orography'         ] = construct_profil_satellite(ncfile_GOCCP, calipso_track, 'Surface_Elevation_Statistics','calipso')
    #profil_calipso['total_cloud_cover' ] = construct_profil_satellite(ncfile_GOCCP, calipso_track, 'Cloud_Layer_Fraction',        'calipso') 

   
    profil_GOCCP = {}
    profil_GOCCP['total_cloud_cover'] = construct_profil_2Ddata(ncfile_GOCCP , indx_gem , GOCCPvar)

    profil_cospout = {}
    profil_cospout['orography'        ] = construct_profil_2Ddata(ncfile_cospin , indx_cospin , 'orography')
    profil_cospout['landmask'         ] = construct_profil_2Ddata(ncfile_cospin , indx_cospin , 'landmask' )
    profil_cospout['total_cloud_cover'] = construct_profil_2Ddata(ncfile_cospout, indx_cospout, cospoutvar)
    profil_cospout['total_cloud_cover'][profil_cospout['total_cloud_cover' ]<0] = 0  #No cloud = 0%

    #profil_gem = {}
    #profil_gem['orography'        ] = profil_cospout['orography']
    #profil_gem['total_cloud_cover'] = construct_profil_2Ddata(ncfile_gem, indx_gem, 'FN', t_gem)



    



    # INTERPOLATION OF VERTICAL LAYERS
    layer_gem     = construct_model_layer(ncfile_cospin, indx_cospin)
    layer_cospout = np.arange(   0,(320+1)*60,60)
    layer_calipso = np.arange(-480,-480+(398+1)*60,60)
    layer_calipso = np.arange(0,0+(39+1)*480,480)
    layer_target  = np.arange(   0,(320+1)*60,60)
    

    overlap_coeff_cospout = compute_overlap_coeff(np.flipud(layer_cospout), np.flip(layer_target))
    overlap_coeff_calipso = compute_overlap_coeff(np.flipud(layer_calipso), np.flip(layer_target))
    #overlap_coeff_gem     = compute_overlap_coeff(np.flipud(layer_gem    ), np.flip(layer_target))

    PROFIL_cospout = {}
    PROFIL_cospout['total_cloud_cover'] = format_levels(profil_cospout['total_cloud_cover'], overlap_coeff_cospout,320)
    PROFIL_cospout['orography'        ] = profil_cospout['orography']
    PROFIL_cospout['landmask'         ] = profil_cospout['landmask']
    mask_orography(PROFIL_cospout,layer_target,'total_cloud_cover')
    mask_sealand  (PROFIL_cospout, masktype ,'total_cloud_cover')
    
    #PROFIL_gem     = {}
    #PROFIL_gem['total_cloud_cover'] = format_levels(profil_gem['total_cloud_cover']    , overlap_coeff_gem    ,320)
    #PROFIL_gem['orography'        ] = profil_gem    ['orography']
    #PROFIL_gem['landmask'         ] = profil_cospout['landmask' ] #using the same sealandmask for 3 datatset
    #mask_orography(PROFIL_gem,layer_target,'total_cloud_cover')
    #mask_sealand  (PROFIL_gem, masktype ,'total_cloud_cover')


    PROFIL_calipso = {}
    PROFIL_calipso['total_cloud_cover'] = format_levels(profil_GOCCP['total_cloud_cover'], overlap_coeff_calipso,320)
    #PROFIL_calipso['orography'        ] = profil_calipso['orography']
    PROFIL_calipso['landmask'         ] = profil_cospout['landmask' ] #using the same sealandmask for 3 datatset
    #mask_orography(PROFIL_calipso,layer_target,'total_cloud_cover')
    mask_sealand  (PROFIL_calipso, masktype ,'total_cloud_cover')

    if i == 0:
        #PROFIL_gemX    ['total_cloud_cover'] = np.copy(PROFIL_gem    ['total_cloud_cover'])
        PROFIL_cospoutX['total_cloud_cover'] = np.copy(PROFIL_cospout['total_cloud_cover'])
        PROFIL_calipsoX['total_cloud_cover'] = np.copy(PROFIL_calipso['total_cloud_cover'])
        #PROFIL_gemX    ['orography'        ] = np.copy(PROFIL_gem    ['orography'        ])
        PROFIL_cospoutX['orography'        ] = np.copy(PROFIL_cospout['orography'        ])
        #PROFIL_calipsoX['orography'        ] = np.copy(PROFIL_calipso['orography'        ])

    else:
        #PROFIL_gemX    ['total_cloud_cover'] = np.append(PROFIL_gemX    ['total_cloud_cover'], PROFIL_gem    ['total_cloud_cover'],axis=1)
        PROFIL_cospoutX['total_cloud_cover'] = np.append(PROFIL_cospoutX['total_cloud_cover'], PROFIL_cospout['total_cloud_cover'],axis=1)
        PROFIL_calipsoX['total_cloud_cover'] = np.append(PROFIL_calipsoX['total_cloud_cover'], PROFIL_calipso['total_cloud_cover'],axis=1)
        #PROFIL_gemX    ['orography'        ] = np.append(PROFIL_gemX    ['orography'        ], PROFIL_gem    ['orography'        ],axis=0)
        PROFIL_cospoutX['orography'        ] = np.append(PROFIL_cospoutX['orography'        ], PROFIL_cospout['orography'        ],axis=0)
        #PROFIL_calipsoX['orography'        ] = np.append(PROFIL_calipsoX['orography'        ], PROFIL_calipso['orography'        ],axis=0)


    print(PROFIL_calipsoX['total_cloud_cover'].shape,PROFIL_calipso['total_cloud_cover'].shape)
    #print(PROFIL_calipsoX['orography'        ].shape,PROFIL_calipso['orography'        ].shape)



# INTENSITY-FREQUENCY FIGURES
for threshold in [0,0.03333333333]:
    th       = str(threshold).replace('.','p')
    data    = {'calipso':  PROFIL_calipsoX, 'cospout':  PROFIL_cospoutX}
    #data    = {'calipso':  PROFIL_calipsoX, 'cospout':  PROFIL_cospoutX, 'gem':  PROFIL_gemX}
    print(dirfig + '/freq_int/' + 'cloud_cover_total_xxxYYY_' +  date  + '_' + maskfname + '_' + s +  '_calipso_COSP2_GEM5_' + th )
    img_att = {'masktype':masktype , 'vstruct':'calipso', 'fignamex': dirfig + '/freq_int/' + 'cloud_cover_total_xxxYYY_' +  date  + '_' + maskfname + '_' + s +  '_calipso_COSP2_GEM5_' + th }
    plot_frequency_intensity_profil(data, 'total_cloud_cover', threshold, img_att)





# CLOUD PROFILE FIGURES
#image_attribute_calipso  = {'title':'CALISPO: Total cloud cover'           , 'vstruct':'calipso', 'fignamex': dirfig + '/profile/' + 'cloud_cover' + '_' + date + '_' + s +  '_calipso'}
#image_attribute_cosp     = {'title':'COSP: Total cloud cover (CALIPSO) '   , 'vstruct':'calipso', 'fignamex': dirfig + '/profile/' + 'cloud_cover' + '_' + date + '_' + s +  '_COSP2'  }
#image_attribute_gem      = {'title':'GEM: Total cloud cover '              , 'vstruct':'calipso', 'fignamex': dirfig + '/profile/' + 'cloud_cover' + '_' + date + '_' + s +  '_GEM5'   }

#plot_profil(PROFIL_calipsoX,'total_cloud_cover', image_attribute_calipso)
#plot_profil(PROFIL_cospoutX,'total_cloud_cover', image_attribute_cosp)
#plot_profil(PROFIL_gemX    ,'total_cloud_cover', image_attribute_gem )

    

# SATELLITE TRACK FIGURE
image_attribute_map = {'fignamex': dirfig + '/../track/cloudsat_' +  str(df['ti'][i]) + '_' + str(df['tf'][i])}
plot_borders_and_tracks(calipso_trackX, grid_cosp,image_attribute_map)


plt.show()





