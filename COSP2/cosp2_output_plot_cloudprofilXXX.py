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



if 1 == 0:
    datex = np.empty(26, dtype=object)
    datax = np.empty(26, dtype=object)
    tx    = np.empty(26, dtype=int)
    i=0
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-08T05-34-42ZN.nc';  datex[i] = '201409080600'; tx[i]= 6; i = i + 1 #
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-08T07-13-32ZN.nc';  datex[i] = '201409080900'; tx[i]= 9; i = i + 1 # 2014-09-08T07:19 2014-09-08T07:26
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-08T08-52-27ZN.nc';  datex[i] = '201409080900'; tx[i]= 9; i = i + 1 # 2014-09-08T08:57 2014-09-08T09:03
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-08T15-27-53ZN.nc';  datex[i] = '201409081700'; tx[i]=17; i = i + 1 # 2014-09-08T16:52 2014-09-08T16:57
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-08T16-14-18ZD.nc';  datex[i] = '201409081700'; tx[i]=17; i = i + 1 # 2014-09-08T16:52 2014-09-08T16:57
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-08T17-06-48ZN.nc';  datex[i] = '201409081900'; tx[i]=19; i = i + 1 # 2014-09-08T18:28 2014-09-08T18:37
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-08T17-53-13ZD.nc';  datex[i] = '201409081900'; tx[i]=19; i = i + 1 # 2014-09-08T18:28 2014-09-08T18:37
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-08T18-45-38ZN.nc';  datex[i] = '201409082000'; tx[i]=20; i = i + 1 # 2014-09-08T20:08 2014-09-08T20:09
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-08T19-32-03ZD.nc';  datex[i] = '201409082000'; tx[i]=20; i = i + 1 # 2014-09-08T20:08 2014-09-08T20:09
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-09T06-17-50ZN.nc';  datex[i] = '201409090600'; tx[i]= 6; i = i + 1 # 2014-09-09T06:24 2014-09-09T06:31
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-09T07-56-40ZN.nc';  datex[i] = '201409090800'; tx[i]= 8; i = i + 1 # 2014-09-09T08:01 2014-09-09T08:09
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-09T09-35-35ZN.nc';  datex[i] = '201409091000'; tx[i]=10; i = i + 1 # 2014-09-09T09:40 2014-09-09T09:41
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-09T14-32-11ZN.nc';  datex[i] = '201409091600'; tx[i]=16; i = i + 1 # 2014-09-09T15:59 2014-09-09T16:00
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-09T15-18-36ZD.nc';  datex[i] = '201409091600'; tx[i]=16; i = i + 1 # 2014-09-09T15:59 2014-09-09T16:00
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-09T16-11-01ZN.nc';  datex[i] = '201409091800'; tx[i]=18; i = i + 1 # 2014-09-09T17:32 2014-09-09T17:41
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-09T16-57-26ZD.nc';  datex[i] = '201409091800'; tx[i]=18; i = i + 1 # 2014-09-09T17:32 2014-09-09T17:41
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-09T17-49-56ZN.nc';  datex[i] = '201409091900'; tx[i]=19; i = i + 1 # 2014-09-09T19:12 2014-09-09T19:20
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-09T18-36-21ZD.nc';  datex[i] = '201409091900'; tx[i]=19; i = i + 1 # 2014-09-09T19:12 2014-09-09T19:20
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-10T15-15-19ZN.nc';  datex[i] = '201409101700'; tx[i]=17; i = i + 1 # 2014-09-10T16:40 2014-09-10T16:44
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-10T16-01-44ZD.nc';  datex[i] = '201409101700'; tx[i]=17; i = i + 1 # 2014-09-10T16:40 2014-09-10T16:44
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-10T16-54-09ZN.nc';  datex[i] = '201409101800'; tx[i]=18; i = i + 1 # 2014-09-10T18:15 2014-09-10T18:24
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-10T17-40-34ZD.nc';  datex[i] = '201409101800'; tx[i]=18; i = i + 1 # 2014-09-10T18:15 2014-09-10T18:24
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-10T18-33-04ZN.nc';  datex[i] = '201409102000'; tx[i]=20; i = i + 1 # 2014-09-10T19:55 2014-09-10T19:59
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-10T19-19-30ZD.nc';  datex[i] = '201409102000'; tx[i]=20; i = i + 1 # 2014-09-10T19:55 2014-09-10T19:59
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-11T06-05-11ZN.nc';  datex[i] = '201409110600'; tx[i]= 6; i = i + 1 # 2014-09-11T06:11 2014-09-11T06:19
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-11T07-44-06ZN.nc';  datex[i] = '201409110800'; tx[i]= 8; i = i + 1 # 2014-09-11T07:49 2014-09-11T07:56
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-15T15-33-27ZN.nc';   2014-09-15T15:33 2014-09-15T17:12   2014-09-15T16:57 2014-09-15T17:03
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-15T16-19-47ZD.nc';   2014-09-15T16:19 2014-09-15T17:12   2014-09-15T16:57 2014-09-15T17:03
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-15T17-12-17ZN.nc';   2014-09-15T17:12 2014-09-15T18:51   2014-09-15T18:34 2014-09-15T18:43
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-15T17-58-42ZD.nc';   2014-09-15T17:58 2014-09-15T18:51   2014-09-15T18:34 2014-09-15T18:43
else:

    datex = np.empty(6, dtype=object)
    datax = np.empty(6, dtype=object)
    tx    = np.empty(6, dtype=int)
    i=0
    datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-09-01T17-47-56ZD.nc';  datex[i] = '201409011800'; tx[i]= 18; i = i + 1
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-01-05T06-03-15ZN.nc';  datex[i] = '201401050600'; tx[i]= 6; i = i + 1
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-01-05T07-42-11ZN.nc';  datex[i] = '201401050800'; tx[i]= 8; i = i + 1
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-01-05T09-21-06ZN.nc';  datex[i] = '201401051000'; tx[i]=10; i = i + 1
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-01-05T15-04-07ZD.nc';  datex[i] = '201401051600'; tx[i]=16; i = i + 1
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-01-05T16-43-02ZD.nc';  datex[i] = '201401051800'; tx[i]=18; i = i + 1
    #datax[i] = 'CAL_LID_L2_05kmCPro-Standard-V4-20.2014-01-05T18-21-52ZD.nc';  datex[i] = '201401051900'; tx[i]=19; i = i + 1

PROFIL_gemX={}
PROFIL_cospoutX={}
PROFIL_calipsoX={}
for i in range(i):
    print('==================',i,datax[i])
    MP='MPB'
    MSC='1M002SC'
    date            = datex[i]
    grid_gem        ='NAM-11m'
    grid_cosp       ='box11'
    gempath         ='Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes'
    dircosp         ='/pampa/poitras/DATA/COSP2/' + gempath
    ncfile_calipso  ='/pampa/poitras/DATA/CALIPSO/CAL_LID_L2_05kmCPro-Standard-V4/NetCDF/2014/' + datax[i]
    ncfile_cospout  = dircosp + '/' + MP + '/OUTPUT/' + MSC + '/cosp_output_' + datex[i] + '_calipso_cloudprofile_2D.nc'
    ncfile_cospin   = dircosp + '/' + MP + '/INPUT/NAM11/2014/cosp_input_' + datex[i] + '.nc'
    ncfile_gem      ='/pampa/poitras/DATA/GEM5/COSP2/' + gempath + '/Samples_NetCDF/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes_201401/pm2013010100_20140105d.nc'
    dirfig          ='/pampa/poitras/figures/COSP2/' +  gempath   + '/' + MP + '/OUTPUT/' + MSC


    timeslice_gem = tx[i]-1


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
    print(ncfile_cospout)
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

    if i == 0:
        PROFIL_gemX    ['total_cloud_cover'] = np.copy(PROFIL_gem    ['total_cloud_cover'])
        PROFIL_cospoutX['total_cloud_cover'] = np.copy(PROFIL_cospout['total_cloud_cover'])
        PROFIL_calipsoX['total_cloud_cover'] = np.copy(PROFIL_calipso['total_cloud_cover'])
        PROFIL_gemX    ['orography'        ] = np.copy(PROFIL_gem    ['orography'        ])
        PROFIL_cospoutX['orography'        ] = np.copy(PROFIL_cospout['orography'        ])
        PROFIL_calipsoX['orography'        ] = np.copy(PROFIL_calipso['orography'        ])

    else:
        PROFIL_gemX    ['total_cloud_cover'] = np.append(PROFIL_gemX    ['total_cloud_cover'], PROFIL_gem    ['total_cloud_cover'],axis=1)
        PROFIL_cospoutX['total_cloud_cover'] = np.append(PROFIL_cospoutX['total_cloud_cover'], PROFIL_cospout['total_cloud_cover'],axis=1)
        PROFIL_calipsoX['total_cloud_cover'] = np.append(PROFIL_calipsoX['total_cloud_cover'], PROFIL_calipso['total_cloud_cover'],axis=1)
        PROFIL_gemX    ['orography'        ] = np.append(PROFIL_gemX    ['orography'        ], PROFIL_gem    ['orography'        ],axis=0)
        PROFIL_cospoutX['orography'        ] = np.append(PROFIL_cospoutX['orography'        ], PROFIL_cospout['orography'        ],axis=0)
        PROFIL_calipsoX['orography'        ] = np.append(PROFIL_calipsoX['orography'        ], PROFIL_calipso['orography'        ],axis=0)


    print(PROFIL_calipsoX['total_cloud_cover'].shape,PROFIL_calipso['total_cloud_cover'].shape)
    print(PROFIL_calipsoX['orography'        ].shape,PROFIL_calipso['orography'        ].shape)



# INTENSITY-FREQUENCY FIGURES

#for threshold in np.arange(0,1.05,0.05):
for threshold in [0,1e-6,0.05]:
    th       = str(threshold).replace('.','p')
    data    = {'calipso':  PROFIL_calipsoX, 'cospout':  PROFIL_cospoutX}
    data    = {'calipso':  PROFIL_calipsoX, 'cospout':  PROFIL_cospoutX, 'gem':  PROFIL_gemX}
    img_att = {'title':'Total cloud cover ' , 'vstruct':'calipso', 'figname': dirfig + '/freq_int/' + 'cloud_cover_total_xxxYYY_' + th + '_' + date + 'X_calipso_COSP2_GEM5'   }
    plot_frequency_intensity_profil(data, 'total_cloud_cover', threshold, img_att)

#plt.show()
#exit()



image_attribute_calipso  = {'title':'CALISPO: Total cloud cover'           , 'vstruct':'calipso', 'figname': dirfig + '/profile/' + 'cloud_cover' + '_' + date + 'X_calipso'}
image_attribute_cosp     = {'title':'COSP: Total cloud cover (CALIPSO) '   , 'vstruct':'calipso', 'figname': dirfig + '/profile/' + 'cloud_cover' + '_' + date + 'X_COSP2'  }
image_attribute_gem      = {'title':'GEM: Total cloud cover '              , 'vstruct':'calipso', 'figname': dirfig + '/profile/' + 'cloud_cover' + '_' + date + 'X_GEM5'   }

plot_profil(PROFIL_calipsoX,'total_cloud_cover', image_attribute_calipso)
plot_profil(PROFIL_cospoutX,'total_cloud_cover', image_attribute_cosp)
plot_profil(PROFIL_gemX    ,'total_cloud_cover', image_attribute_gem )

    #plt.figure(4)
    #plt.plot(profil_calipso['orography'],'r-')
    #plt.plot(profil_cospout['orography'        ],'b-')
plt.show()





