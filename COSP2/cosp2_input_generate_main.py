# info: poitras.vincent@uqam.ca
# date: 2022/02/18
# aim : convert gem output into cosp input
# Note: A part of this script is adapted from previous work made by Zhipeng Qu.
#       Discussions from our "COSP group" (Faisal Boudala, Mélissa Cholette, Jason Milbrant,Vincent Poitras, Zhipeng Qu)
#       also help to develop this script.
from sys import getsizeof
import numpy             as     np
import matplotlib.pyplot as     plt
from   datetime          import datetime
import netCDF4
import os

import sys;                       sys.path.append('/home/poitras/SCRIPTS/mes_modules_python')
from   netcdf4_extra              import netcdf4_extract_fields_and_attributes
from   cosp2_input_generate_module import add_time_dimension
from   cosp2_input_generate_module import format_time_for_sunlit
from   cosp2_input_generate_module import sunlit
from   cosp2_input_generate_module import radius_from_cldoppro
from   cosp2_input_generate_module import radius_from_mp_my2
from   cosp2_input_generate_module import cloud_optical_depth_and_emmissivity
from   cosp2_input_generate_module import write_ncfile_output

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

#############################################################################################################################
#                                                   I/O PATH FILES (to edit)                                                #
#############################################################################################################################
# The input data were converted from fst to NetCDF format using
#   python -m  fstd2nc --nc-format NETCDF4 --zlib --keep-LA-LO $filei $fileo
#   If you use the same command, the fields will have the correct name to be used in this script   


#MP_CONFIG='MPB'
#diri  = '/pampa/poitras/DATA/TREATED/GEM5/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/storm_019/Samples_NetCDF_COSP'
#diri0 = '/pampa/poitras/DATA/TREATED/GEM5/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/storm_019~/Samples_NetCDF_COSP'
#ncfiles_pm0 = [ diri0 + '/storm_019_step0/pm2014010100_00000000p.nc' ] # FOR MG
#ncfiles_dm0 = [ diri0 + '/storm_019_step0/dm2014010100_00000000p.nc' ] # FOR ME
#ncfiles_pm  = [ diri  + '/storm_019_201401/pm2013010100_20140105d.nc']
#ncfiles_dm  = [ diri  + '/storm_019_201401/dm2013010100_20140105d.nc']
#dirout      = '/pampa/poitras/DATA/TREATED/COSP2/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/' + MP_CONFIG + '/INPUT'



ncfiles_pm0 = [ sys.argv[1] ]
ncfiles_dm0 = [ sys.argv[2] ]
ncfiles_pm  = [ sys.argv[3] ]
ncfiles_dm  = [ sys.argv[4] ]
dirout      =   sys.argv[5]
MP_CONFIG   =   sys.argv[6]



if not os.path.exists(dirout): os.makedirs(dirout)





# To test this rapidely test this script, you can uncomment the following lines:
#ncfiles_pm = ncfiles_pm0 
#ncfiles_dm = ncfiles_dm0

#############################################################################################################################
#                                                     READING INPUT FILE                                                    #
#############################################################################################################################
#   Note: The script "netcdf4_extract_fields_and_attributes" is used to read data and attributes from the ncfiles
#         This scripts was developped for an other project and might be unnecessarily complicated here, but it work perfectly

print('Reading input data')
 
#varlist_pm0 = ['MG']
#varlist_dm0 = ['ME']
#varlist_pm  = ['FMP', 'FN', 'J8', 'QI_1', 'QI_2', 'QI_3', 'QI_4', 'QI_5', 'QI_6', 'REC', 'REI1', 'LWCI', 'IWCI', 'SS01']
#varlist_dm  = ['HU', 'TT','MPQC','MPQR','MPNR', 'GZ', 'P0','leadtime', 'a_1', 'b_1', 'a_2', 'b_2', 'pref' ,'reftime','lon','lat','rlon','rlat' ]

#input_data       = {}; 
#input_attributes = {};
#netcdf4_extract_fields_and_attributes(varlist_pm0, ncfiles_pm0, input_data, input_attributes);
#netcdf4_extract_fields_and_attributes(varlist_dm0, ncfiles_dm0, input_data, input_attributes);
#netcdf4_extract_fields_and_attributes(varlist_pm , ncfiles_pm , input_data, input_attributes);
#netcdf4_extract_fields_and_attributes(varlist_dm , ncfiles_dm , input_data, input_attributes);


# Extract dimension size
varlist_dm  = [ 'TT']

input_data       = {}; 
input_attributes = {};
netcdf4_extract_fields_and_attributes(varlist_dm , ncfiles_dm , input_data, input_attributes);


XX = input_data['TT'].copy()
if   len(XX.shape) == 3:        # [      vertical level, lat, lon]
    XX = XX [np.newaxis,:,:,:]  # [time, vertical level, lat, lon]

ntime = XX.shape[0]
nlev  = XX.shape[1]
nlat  = XX.shape[2]
nlon  = XX.shape[3]

del XX

# Adding a time dimension (for single step ncfile)
#   If the NetCDF file contain only one time step (ntime == 1), there will be no time dimension in the input field
#   We are adding it because the rest of the script is expecting to have one
#        [                        ] --> [time,                         ] : time
#        [                lat, lon] --> [time,                 lat, lon] : P0
#        [ surfacei_type, lat, lon] --> [time,   surface_type, lat, lon] : J8
#        [vertical_level, lat, lon] --> [time, vertical_level, lat, lon] : Other variables
#if ntime == 1:
#    # variable such as lon, lat etc. are time independent, so they are remoevd from the list below
#    varlist = list(set(varlist_pm + varlist_dm) - set(['a_1', 'b_1', 'a_2', 'b_2', 'pref' ,'reftime','lon','lat','rlon','rlat'])) 
#    add_time_dimension(varlist,input_data)                                                            
    

#############################################################################################################################
#                                                       Creation of the outputfile                                          #
#############################################################################################################################
# Creating the output files at the begonning of the script to be able to append results at each steps and avoid memeory issue


varlist_dm  = ['leadtime' ,'reftime','rlon','rlat' ]
output_data      = {}
input_data       = {}; 
input_attributes = {};
netcdf4_extract_fields_and_attributes(varlist_dm , ncfiles_dm , input_data, input_attributes);




# Space and time coordinate:
output_data['lon'] = input_data['rlon'    ];  ### rotated longitude
output_data['lat'] = input_data['rlat'    ];  ### rotated latitude
time               = input_data['leadtime'];
reftime            = input_attributes['reftime']['units'];
formatted_time     = format_time_for_sunlit(reftime, time)
input_data.pop('rlon')
input_data.pop('rlat')
input_data.pop('leadtime')


print('Creating output file')
for t in range(ntime):

    # Create filename (they will be labelled by the date + hour)
    time       = datetime.strptime(formatted_time[t],'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')
    ncfile_path = dirout + '/cosp_input_' + time + '.nc'
    ncfile      = netCDF4.Dataset(ncfile_path, 'w', format='NETCDF4')

    # Create dimension
    ncfile.createDimension('lon'  ,nlon)
    ncfile.createDimension('lat'  ,nlat)
    ncfile.createDimension('level',nlev)
    ncfile.createDimension('hydro',   9)

    # Create variable: lat, lon, emsfc_lw
    ncfile.createVariable('emsfc_lw', 'f4')
    ncfile['emsfc_lw'][:] = 0.95
    ncfile.createVariable('lon', 'f4', ('lon'))
    ncfile['lon'][:] = output_data['lon']
    ncfile.createVariable('lat', 'f4', ('lat'))
    ncfile['lat'][:] = output_data['lat']



output_data['ap_lsliq'] = np.ones((ntime, nlev, nlat, nlon), dtype='float32') * 524.00
output_data['ap_lsice'] = np.ones((ntime, nlev, nlat, nlon), dtype='float32') * 110.80
output_data['bp_lsliq'] = np.ones((ntime, nlev, nlat, nlon), dtype='float32') *   3.00
output_data['bp_lsice'] = np.ones((ntime, nlev, nlat, nlon), dtype='float32') *   2.91
write_ncfile_output(output_data,formatted_time,dirout)





#############################################################################################################################
#                                         Miscalenous fields that can be directly copied                                    #
#############################################################################################################################
# 5 mixing ratio are also directly copied (see the corresponding section below)
# Note on the lat/lon fields:
#   As far I understand, only rttov is actually using lat/lon. Otherwise lat/lon is just used as label for the indices,
#   so there is nothing wrong using a rotated grid with rlat/rlon
#   But we still to use the "real" lat/lon field to
#       [1] compute sunlit
#       [2] use as reference to make sure that everything is correct


varlist_dm0 = ['ME']
varlist_pm  = ['FMP', 'FN',]
varlist_dm  = ['HU' ,'lon','lat' ]

output_data      = {}
input_data       = {}
input_attributes = {}
netcdf4_extract_fields_and_attributes(varlist_dm0, ncfiles_dm0, input_data, input_attributes);
netcdf4_extract_fields_and_attributes(varlist_pm , ncfiles_pm , input_data, input_attributes);
netcdf4_extract_fields_and_attributes(varlist_dm , ncfiles_dm , input_data, input_attributes);

output_data['orography'] = input_data['ME'  ];  ### Topography
output_data['tca'      ] = input_data['FN'  ];  ### Total cloud fractions
output_data['cca'      ] = input_data['FMP' ];  ### Implicit cloud fractions
output_data['q_v'      ] = input_data['HU'  ];  ### Specifi humidity
output_data['longitude'] = input_data['lon' ];  ### longitude (unrotated grid)
output_data['latitude' ] = input_data['lat' ];  ### latitude  (unrotated grid)

orography = input_data['ME'  ]
lon       = input_data['lon'];
lat       = input_data['lat'];

write_ncfile_output(output_data,formatted_time,dirout)







#############################################################################################################################
#                                                Full-level and half-level height                                           #
#############################################################################################################################
# Dimensions (multiple step): (time, vertical level, x, y)
# GZ is given in Decametre
# We use thermodynamic levels as full height
# We use momentum      levels as half height
# Note that the order is "inverted":
#   nGZ-1 is the 1st level above the ground
#   0     is the TOA    
# momentum      levels (half): 1, 3, 5, ...,  nGZ-5, nGZ-3, nGZ-1
# thermodynamic levels (full): 0, 1, 4, ...,  nGZ-6, nGZ-4, nGZ-2
# There is no level nGZ (surface, hyb = 1) since this field is equivalent to the topography


varlist_dm  = [ 'GZ' ]
output_data      = {};
input_data       = {};
input_attributes = {};
netcdf4_extract_fields_and_attributes(varlist_dm , ncfiles_dm , input_data, input_attributes);

GZ  = 10 * input_data['GZ'] # Dm -> m
nGZ = len(GZ[0,:,0,0])

half_level_indices = range(1, nGZ  ,  2) # 1 to nGZ-1
full_level_indices = range(0, nGZ-1,  2) # 0 to nGZ-2

half_height = GZ[:,half_level_indices,:,:]
full_height = GZ[:,full_level_indices,:,:]
output_data['height']      = full_height;  ### Height_in_full_levels
output_data['height_half'] = half_height;  ### Height_in_half_levels

write_ncfile_output(output_data,formatted_time,dirout)

del GZ, nGZ, half_height 

#############################################################################################################################
#                                                Full-pressure and half-pressure level                                      #
#############################################################################################################################
# For half/full levels explanation, see the section on full-level and half-level height above  


varlist_dm  = [ 'P0', 'a_1', 'b_1', 'a_2', 'b_2', 'pref' ]
output_data      = {};
input_data       = {};
input_attributes = {};
netcdf4_extract_fields_and_attributes(varlist_dm , ncfiles_dm , input_data, input_attributes);

a_x    = 'a_1' if len(input_data['a_1']) > len(input_data['a_2']) else 'a_2' 
b_x    = 'b_1' if len(input_data['b_1']) > len(input_data['b_2']) else 'b_2'

a    = input_data[ a_x  ]
b    = input_data[ b_x  ]
pref = input_data['pref']       # Reference pressure
ps   = input_data['P0'  ] * 100 # Surface pressure [hPa --> Pa]


p = np.zeros((ntime,2*nlev,nlat,nlon), dtype='float32')
for i in range(2*nlev):
    p[:,i,:,:]  =  np.exp(a[i]+b[i]*np.log(ps/pref))

phalf = p[:,half_level_indices,:,:]
pfull = p[:,full_level_indices,:,:]

output_data['pfull'] = pfull;  ### pressure_in_full_levels
output_data['phalf'] = phalf;  ### pressure_in_half_levels
write_ncfile_output(output_data,formatted_time,dirout)


del a, b, pref, ps, p, phalf 


#############################################################################################################################
#                                                           Sunlit                                                          #
#############################################################################################################################
# We set daylight (sunlit=1) when solar zenith < 90
#lon     = input_data['lon'];
#lat     = input_data['lat'];
#time    = input_data['leadtime'];
#reftime = input_attributes['reftime']['units'];
#formatted_time = format_time_for_sunlit(reftime, time)

output_data['sunlit']         = sunlit(lon,lat,formatted_time)

#output_data['sunlit'] = sunlit; ### Daypoints 
write_ncfile_output(output_data,formatted_time,dirout)

del lon, lat

#############################################################################################################################
#                                                           Sea-land mask                                                   #
#############################################################################################################################
#   In gem. the sea-land mask varies between 0 (ony sea) and 1 (only land)
#   COSP requieres a binary mask
#   To be consistant with cldoppro_mp.F90, we do: mask <= 0.5 --> 0 (only sea), mask > 0.5 --> (only land)


varlist_pm0 = ['MG']
output_data      = {};
input_data       = {};
input_attributes = {};
netcdf4_extract_fields_and_attributes(varlist_pm0, ncfiles_pm0, input_data, input_attributes);


landmask = input_data['MG']
landmask [ landmask <= 0.5] = 0
landmask [ landmask >  0.5] = 1 

output_data['landmask'] = landmask;  ### Sea-land mask
write_ncfile_output(output_data,formatted_time,dirout)



#############################################################################################################################
#                                                       Skin temperature                                                    #
#############################################################################################################################
# GEM output skin temperature over 7 surface fractions:
# 0: soil, 1: glacier, 2: sea-water, 3: sea-ice, 4: aggregated, 5: urban, 6: lake
# We are interested in the aggregated value (4)

varlist_pm  = ['J8']
output_data      = {};
input_data       = {};
input_attributes = {};
netcdf4_extract_fields_and_attributes(varlist_pm , ncfiles_pm , input_data, input_attributes);


J8 = input_data['J8'][:,4,:,:];
output_data['skt'] = J8;  ### Skin temperature

write_ncfile_output(output_data,formatted_time,dirout)

del J8

#############################################################################################################################
#                                                         Air temperature                                                   #
#############################################################################################################################
#   Converting from celsius to kelvin

varlist_dm  = ['TT' ]

output_data      = {};
input_data       = {};
input_attributes = {};
netcdf4_extract_fields_and_attributes(varlist_dm , ncfiles_dm , input_data, input_attributes);




T_abs = input_data['TT'] + 273.15;
output_data['T_abs'] = T_abs; ### Temperature

write_ncfile_output(output_data,formatted_time,dirout)



#############################################################################################################################
#                                                         Mixing ratio                                                      #
#############################################################################################################################

varlist_pm  = ['QI_1', 'QI_2', 'QI_3', 'QI_4', 'QI_5', 'QI_6', 'LWCI', 'IWCI']
varlist_dm  = ['MPQC','MPQR','MPNR' ]

output_data      = {};
input_data       = {};
input_attributes = {};
netcdf4_extract_fields_and_attributes(varlist_pm , ncfiles_pm , input_data, input_attributes);
netcdf4_extract_fields_and_attributes(varlist_dm , ncfiles_dm , input_data, input_attributes);



# Mixing ratio 1: Data directly copied

output_data['mr_lsliq' ] = input_data['MPQC'];  ### Mixing ratio: cloud-liquid (large scale)
output_data['mr_lsrain'] = input_data['MPQR'];  ### Mixing ratio: precip-rain  (large scale)
output_data['mr_ccliq' ] = input_data['LWCI'];  ### Mixing ratio: cloud-liquid (convective)  [GEM: PBL + 3 convec schemes]
output_data['mr_ccice' ] = input_data['IWCI'];  ### Mixing ratio: cloud-ice    (convective)  [GEM: PBL + 3 convec schemes]

# Mixing ratio 2: Decomposition of QI_X: --> cloud-ice  (large scale), precip-snow (large scale) and  precip-graupel (large scale)
q1 = input_data['QI_1']  #[GEM: small ice crystals   ]
q2 = input_data['QI_2']  #[GEM: unrimed snow crystals]
q3 = input_data['QI_3']  #[GEM: lightly rimed snow   ]
q4 = input_data['QI_4']  #[GEM: graupel              ]
q5 = input_data['QI_5']  #[GEM: hail                 ]
q6 = input_data['QI_6']  #[GEM: ice pellet           ]

if   MP_CONFIG == 'MPA':
    output_data['mr_lsice' ] = q1          ;  ### Mixing ratio: cloud-ice      (large scale)
    output_data['mr_lssnow'] = q2 + q3     ;  ### Mixing ratio: precip-snow    (large scale)
    output_data['mr_lsgrpl'] = q4 + q5 + q6;  ### Mixing ratio: precip-graupel (large scale)
elif MP_CONFIG == 'MPB':
    output_data['mr_lsice' ] = q1 + q2     ;  ### Mixing ratio: cloud-ice      (large scale)
    output_data['mr_lssnow'] = q3          ;  ### Mixing ratio: precip-snow    (large scale)
    output_data['mr_lsgrpl'] = q4 + q5 + q6;  ### Mixing ratio: precip-graupel (large scale)
elif MP_CONFIG == 'MPC':
    output_data['mr_lsice' ] = (q1 + q2 + q3)/2;  ### Mixing ratio: cloud-ice      (large scale)
    output_data['mr_lssnow'] = (q1 + q2 + q3)/2;  ### Mixing ratio: precip-snow    (large scale)
    output_data['mr_lsgrpl'] = q4 + q5 + q6    ;  ### Mixing ratio: precip-graupel (large scale)


# Mixing ratio 3: precip-snow  (convective) and precip-rain  (convective) 
#   We decided to let the convective_cloud_snow  empty for now
#   Jason will provided a "receipe" for convective_cloud_rain, but untill we have it, we are leaving the field empty
output_data['mr_ccsnow' ] = output_data['mr_lsliq' ] * 0;  ### Mixing ratio: precip-snow  (convective)
output_data['mr_ccrain' ] = output_data['mr_lsliq' ] * 0;  ### Mixing ratio: precip-rain  (convective)


mixing_ratio_tmp = output_data.copy() 
mixing_ratio_tmp.pop('mr_ccrain')
mixing_ratio_tmp.pop('mr_ccsnow')

write_ncfile_output(output_data,formatted_time,dirout)
del q1, q2, q3, q4, q5, q6



#############################################################################################################################
#                                                         Effective radius                                                  #
#############################################################################################################################
varlist_pm  = [ 'REC', 'REI1', 'SS01']
output_data      = {};
input_data       = {};
input_attributes = {};
netcdf4_extract_fields_and_attributes(varlist_pm , ncfiles_pm , input_data, input_attributes);


nhydro = 9
Reff = np.zeros((ntime, nhydro, nlev, nlat, nlon), dtype='float32')

# Effective radius 1: Data directly copied ( Cloud-liquid(LS), Precip-rain(LS))

#print(flag_lsliq.dtype, flag_lsliq.shape)

#for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                         key= lambda x: -x[1])[:10]:
#    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
#my_dictionary = input_data;  size = getsizeof(my_dictionary); size += sum(map(getsizeof, my_dictionary.values())) + sum(map(getsizeof, my_dictionary.keys())); print(size)
#my_dictionary = output_data; size = getsizeof(my_dictionary); size += sum(map(getsizeof, my_dictionary.values())) + sum(map(getsizeof, my_dictionary.keys())); print(size)
#print(input_data)
#print(output_data)

flag_lsliq  = mixing_ratio_tmp['mr_lsliq' ] > 0
flag_lsrain = mixing_ratio_tmp['mr_lsrain'] > 0
mixing_ratio_tmp.pop('mr_lsrain')

Reff[:,0,:,:,:] = input_data['REC' ] * flag_lsliq ;    # Effective radius: cloud-liquid  (large scale)
Reff[:,2,:,:,:] = input_data['SS01'] * flag_lsrain;    # Effective radius: precip-rain   (large scale)



# Effective radius 2:  cloud-ice (large scale), precip-snow (large scale) and precip-graupel (large scale)
#   GEM produced actually making difference between cloud-ice, snow or graupel since it is producing a single mixing ratio QTI1  
#   At each step is diagnostic is done to assign to each griobox a single hydrometeore type category (small ice crystals, unrimed snow crystals, etc.)
#   QI_1 = QTI1 [flag = small ice crystals], QI_2 = QTI2 [flag = unrimed snow crystals], etc
#   This decomposition was not conducted for the effective radius (REI1), we will nthe conduct it here
flag_lsice  = mixing_ratio_tmp['mr_lsice' ] > 0
flag_lssnow = mixing_ratio_tmp['mr_lssnow'] > 0
flag_lsgrpl = mixing_ratio_tmp['mr_lsgrpl'] > 0
mixing_ratio_tmp.pop('mr_lssnow')
mixing_ratio_tmp.pop('mr_lsgrpl')

Reff[:,1,:,:,:]  =  input_data['REI1'] * flag_lsice  # Effective radius: cloud-ice      (large scale)
Reff[:,3,:,:,:]  =  input_data['REI1'] * flag_lssnow # Effective radius: precip-snow    (large scale)
Reff[:,8,:,:,:]  =  input_data['REI1'] * flag_lsgrpl # Effective radius: precip-graupel (large scale)



# Effective radius 3:
#   Using the equations in cldoppro_mp.F90 ro compute effective radius of convective cloud hydrometeores
#   There is 2 options for ice_cloud, but as far I understand the options with a  constant value is used (to check) 
#   When option ice_constant is chosen, r = 15 micron if mixing ratio > 1e-10, 0-->NaN elswhere
RGASD        = 0.28705e+3;             # Gas constant for dry air [J K-1 kg-1]
air_density  = ( pfull/(RGASD*T_abs) ).astype(np.float32)

Reff[:,4,:,:,:] = radius_from_cldoppro(air_density, landmask, mixing_ratio_tmp['mr_ccliq'], 'liquid'      ) # Effective radius: cloud-liquid  (convective)
Reff[:,5,:,:,:] = radius_from_cldoppro(air_density, landmask, mixing_ratio_tmp['mr_ccice'], 'ice_constant') # Effective radius: cloud-ice  (convective)

# Effective radius 4:
#   We are setting the effective radius of convective precipitation to zero
#   For now, we decided to neglect the convective snow
#   For convective rain, we are waiting for Jason "receipe"
Reff[:,6,:,:,:] = Reff[:,4,:,:,:] * 0
Reff[:,7,:,:,:] = Reff[:,4,:,:,:] * 0

# Effective radius 5:
#   WARNING: The equations/parameter taken from mp_my2, seems to apply to
#            frozen precipitation rather than rain --> to discuss
#number_ratio = input_data['MPNR']
#mixing_ratio = input_data['MPQR']
#r_lsrain = radius_from_mp_my2(number_ratio, mixing_ratio, air_density, 'rain')

#Reff[:,0,:,:,:] = r_lsliq    
#Reff[:,1,:,:,:] = r_lsice
#Reff[:,2,:,:,:] = r_lsrain  
#Reff[:,3,:,:,:] = r_lssnow  
#Reff[:,4,:,:,:] = r_ccliq   
#Reff[:,5,:,:,:] = r_ccice
#Reff[:,6,:,:,:] = r_ccrain 
#Reff[:,7,:,:,:] = r_ccsnow 
#Reff[:,8,:,:,:] = r_lsgrpl 

#for i in range(9):
#    print(i,  np.min(np.min(np.min(np.nanmin(Reff[:,i,:,:,:])))) ,  np.max(np.max(np.max(np.nanmax(Reff[:,i,:,:,:])))) )


r_ccliq = Reff[:,4,:,:,:]
r_ccice = Reff[:,5,:,:,:]
output_data['Reff'] = Reff; ### effective radii
write_ncfile_output(output_data,formatted_time,dirout)

del pfull, T_abs, Reff, input_data, flag_lsice, flag_lssnow, flag_lsgrpl, flag_lsliq, flag_lsrain






#############################################################################################################################
#                                               Optical depth and emmisivity                                                #
#############################################################################################################################
# Cloud visible optical depth (0.67 micron) and infrared emmisivity (10.5 micron)
#   Emmisivity and optical depth are computing using equations in clodoppro
#   To be consitent, we are using the radius from clodoppro (instead to REC/REI1 for LS part)


for t in range(ntime):
    time       = datetime.strptime(formatted_time[t],'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')
    ncfile_path = dirout + '/cosp_input_' + time + '.nc'
    ncfile      = netCDF4.Dataset(ncfile_path, 'a', format='NETCDF4')

    output_data = {};

    # Large scale cloud
    mrw = mixing_ratio_tmp['mr_lsliq'][t]; 
    mri = mixing_ratio_tmp['mr_lsice'][t]; 
    rew = radius_from_cldoppro(air_density[t], landmask, mrw, 'liquid'      ) * 1e6 
    rei = radius_from_cldoppro(air_density[t], landmask, mri, 'ice_constant') * 1e6

    print('rew'        ,rew.shape)
    print('rei'        ,rei.shape)
    print('mrw'        ,mrw.shape)
    print('mri'        ,mri.shape)
    print('full_height',full_height.shape)
    print('orography'  ,orography.shape)
    print('air_density',air_density.shape)


    output_data['dtau_s'], output_data['dem_s' ] = cloud_optical_depth_and_emmissivity(rew, rei, mrw, mri, full_height[t][np.newaxis,:,:,:], orography, air_density[t])

    print('dtau_s',output_data['dtau_s'].shape)
    del mrw, mri, rew, rei

    #output_data['dtau_s'] = dtau_s; ###  Cloud Optical depth of at 0.67 micron (visible) (large_scale)
    #output_data['dem_s' ] = dem_s;  ###  Cloud emissivity at 10.5 micron (infrared) (large_scale)
    #write_ncfile_output(output_data,formatted_time,dirout)



    # Convective cloud
    mrw = mixing_ratio_tmp['mr_ccliq'][t]; 
    mri = mixing_ratio_tmp['mr_ccice'][t]; 
    #rew = r_ccliq * 1e6 
    #rei = r_ccice * 1e6
    print('r_ccliq'    ,r_ccliq.shape)
    print('r_ccice'    ,r_ccice.shape)
    print('mrw'        ,mrw.shape)
    print('mri'        ,mri.shape)
    print('full_height',full_height.shape)
    print('orography'  ,orography.shape)
    print('air_density',air_density.shape)
    output_data['dtau_c'], output_data['dem_c' ] = cloud_optical_depth_and_emmissivity(r_ccliq[t] * 1e6, r_ccice[t] * 1e6, mrw, mri, full_height[t][np.newaxis,:,:,:], orography, air_density[t])

    del mrw, mri
    tx=0
    for fieldname in output_data: 
        fieldshape = output_data[fieldname].shape
        print(fieldname,fieldshape)
        #if t == 0 :
            #output_data[fieldname] = np.nan_to_num(output_data[fieldname]) # Set NaN --> 0, doing one time (t=0) for all timestep
    
        if   len(fieldshape) == 2:      # orography, landmask, longitude, latitude
            ncfile.createVariable(fieldname, 'f4', ('lat','lon'))
            ncfile[fieldname][:] = np.nan_to_num( output_data[fieldname] )
    
        elif len(fieldshape) == 3:      # sunlit, skt
            ncfile.createVariable(fieldname, 'f4', ('lat','lon'))
            ncfile[fieldname][:] = np.nan_to_num( output_data[fieldname][tx,:,:] )
    
        elif len(fieldshape) == 4:      # all other variables
            ncfile.createVariable(fieldname, 'f4', ('level','lat','lon'))
            ncfile[fieldname][:] = np.nan_to_num( np.flip(output_data[fieldname][tx,:,:,:],0) )
    
        elif len(fieldshape) == 5:      # effective radii
            ncfile.createVariable(fieldname, 'f4', ('hydro','level','lat','lon'))
            ncfile[fieldname][:] = np.nan_to_num( np.flip(output_data[fieldname][tx,:,:,:,:],1) )


    #output_data['dtau_c'] = dtau_c; ###  Cloud Optical depth of at 0.67 micron (visible) (convective)
    #output_data['dem_c' ] = dem_c;  ###  Cloud emissivity at 10.5 micron (infrared) (convective)

    #write_ncfile_output(output_data,formatted_time,dirout)
    #output_data      = {};
    #input_data       = {};
    #input_attributes = {};




#output_data['dem_c' ]    = output_data['dem_c' ] * 0
#output_data['dtau_c']    = output_data['dtau_c'] * 0
#output_data['cca'      ] = output_data['cca'      ] * 0
#output_data['mr_ccliq' ] = output_data['mr_ccliq' ] * 0
#output_data['mr_ccice' ] = output_data['mr_ccice' ] * 0 
#output_data['mr_ccsnow'] = output_data['mr_ccsnow'] * 0
#output_data['mr_ccrain'] = output_data['mr_ccrain'] * 0
#Reff[:,4,:,:,:] = r_ccliq  * 0
#Reff[:,5,:,:,:] = r_ccice  * 0
#Reff[:,6,:,:,:] = r_ccrain * 0
#Reff[:,7,:,:,:] = r_ccsnow * 0
#output_data['Reff'] = Reff




#############################################################################################################################
#                                                   Creating output files                                                   #
#############################################################################################################################
#for field in output_data: print(field)

#print('Creating output file')

for t in range(ntime):
    
    # Create filename (they will be labelled by the date + hour)
    time       = datetime.strptime(formatted_time[t],'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')
    ncfile_path = dirout + '/cosp_input_' + time + '.nc'
    #ncfile      = netCDF4.Dataset(ncfile_path, 'w', format='NETCDF4')
    ncfile      = netCDF4.Dataset(ncfile_path, 'a', format='NETCDF4')

    print(ncfile_path)
    # Create dimension
    #ncfile.createDimension('lon'  ,nlon)
    #ncfile.createDimension('lat'  ,nlat)
    #ncfile.createDimension('level',nlev)
    #ncfile.createDimension('hydro',   9)

    # Create variable: lat, lon, emsfc_lw
    #ncfile.createVariable('emsfc_lw', 'f4')
    #ncfile['emsfc_lw'][:] = 0.95
    #ncfile.createVariable('lon', 'f4', ('lon'))
    #ncfile['lon'][:] = output_data['lon']
    #ncfile.createVariable('lat', 'f4', ('lat'))
    #ncfile['lat'][:] = output_data['lat']
    
    # Cerate other variables
    #for fieldname in output_data:
    #
    #    fieldshape = output_data[fieldname].shape
    #    #if t == 0 :
    #        #output_data[fieldname] = np.nan_to_num(output_data[fieldname]) # Set NaN --> 0, doing one time (t=0) for all timestep
    #
    #    if   len(fieldshape) == 2:      # orography, landmask, longitude, latitude
    #        ncfile.createVariable(fieldname, 'f4', ('lat','lon'))
    #        ncfile[fieldname][:] = np.nan_to_num( output_data[fieldname] )
    #
    #    elif len(fieldshape) == 3:      # sunlit, skt
    #        ncfile.createVariable(fieldname, 'f4', ('lat','lon'))
    #        ncfile[fieldname][:] = np.nan_to_num( output_data[fieldname][t,:,:] )
    #
    #    elif len(fieldshape) == 4:      # all other variables
    #        ncfile.createVariable(fieldname, 'f4', ('level','lat','lon'))
    #        ncfile[fieldname][:] = np.nan_to_num( np.flip(output_data[fieldname][t,:,:,:],0) )
    #
    #    elif len(fieldshape) == 5:      # effective radii
    #        ncfile.createVariable(fieldname, 'f4', ('hydro','level','lat','lon'))
    #        ncfile[fieldname][:] = np.nan_to_num( np.flip(output_data[fieldname][t,:,:,:,:],1) )

    
    # Facultative fields attributes usefull for geolocations (ncview, panoply, etc.)
    ncgrid = ncfiles_pm0[0]
    grd    = netCDF4.Dataset(ncgrid , 'r');
    x      = ncfile.createVariable('rotated_pole', grd['rotated_pole'].datatype )
    ncfile['rotated_pole'].setncatts(grd['rotated_pole'].__dict__)
    ncfile['lat'         ].setncatts(grd['rlon'        ].__dict__)
    ncfile['lon'         ].setncatts(grd['rlat'        ].__dict__)



    print(ncfile_path)

