import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import os
from   shapely.geometry         import Point
from   shapely.geometry.polygon import Polygon
from   datetime                 import datetime
from   datetime                 import timedelta
from   pylab                    import cm


import sys;                      sys.path.append('/home/poitras/SCRIPTS/mes_modules_python')
from   netcdf4_extra             import netcdf4_extract_fields_and_attributes
from   netcdf4_extra         import ncdump
#from   temporal_manipulation import compute_seasmean
from   generate_border       import generate_border
from   grid_projection       import domainbox
from   grid_projection       import read_gem_settings
from   grid_projection       import convert_rotlatlon_to_cartesian as latlon2indx
from   colorbar_extra  import set_colorbar_extend as cbext
NaN = np.nan


figure_directory='/pampa/poitras/figures/COSP'


#########################################################################################################################################################
#                                                                           MODULES                                                                     #
#########################################################################################################################################################
def plot_domain(border, lim,  box1, box2):

    plt.plot(border['i'], border['j'], '-k', linewidth=0.75);
    plt.axis('scaled')
    plt.axis(xmin=lim['xlim'][0], xmax=lim['xlim'][1], ymin=lim['ylim'][0], ymax=lim['ylim'][1]);
    plt.plot(box1[0], box1[1], '-k', linewidth=0.75);
    plt.plot(box2[0], box2[1], '-k', linewidth=0.75);
    plt.xticks([])
    plt.yticks([])

def construct_coord(x,y):
    coord = []
    N     = len(x)
    for n in range(N):
        coord.append((x[n],y[n]))
    return coord

def point_inside_polygon(points, polygon):
    polygon = Polygon(polygon)
    N       = len(points)
    flag    = np.zeros(N)
    for n in range(N):
        point   = Point(points[n])
        if polygon.contains(point):
            flag[n] = 1
    return flag

def select_points(data, flag):
    output = {}
    N = len(flag)
    for var in data:
            output[var] = data[var][flag == 1]
                   
    return output

def set_time_cloudsat(file):
    # Let YYYYMMDDhhmmss.dt be the time of the first timestep
    # YYYYMMDDhhmmss.dt[date]  = YYYYMMDD000000[date] + UTC_start[second]
    # YYYYMMDDhhmmss   [date]  = start_time    [date]
    # Since we don't know a priori YYYYMMDD000000, we do:
    #   YYYYMMDDhhmmss.dt = start_time + millisecond part of UTC_start

    format_in    = '%Y%m%d%H%M%S'
    format_out   = '%Y%m%d%H%M%S.%f'
    nc           = netCDF4.Dataset(file,'r');
    Profile_time = nc['Profile_time'][:]
    N            = len(Profile_time)
    start_time   = datetime.strptime(nc.getncattr('start_time'),format_in)
    UTC_start    = nc['UTC_start'][:]

    dt           = int((nc['UTC_start'][:] % 1) * 1000)  # dt is a integer, in milliseconds
    print(dt)
    start_timedt = start_time + timedelta(milliseconds=dt)  
    time         = np.zeros(N)
    for n in range(N):
        time[n] = (start_timedt + timedelta(milliseconds=Profile_time[n]*1000)).strftime(format_out)
    return time;

def format_data_cs(data,attributes):
    for var in data:
        print(var)
        if 'missing' in attributes[var]:
            missing   = attributes[var]['missing']
            missop    = attributes[var]['missop' ]
            print(missing, missop)
            #data[var] = data[var] 
            if missop == '==':
                data[var] [data[var] == missing ] = NaN
            #else:
            #    print('Missing operator ' + missop + ' not implemented yet')
            #    exit()
        if 'offset' in attributes[var]:
            offset    = attributes[var]['offset']
            factor    = attributes[var]['factor']
            data[var] = (data[var]-offset)/factor


def getOverlap(a, b):
    min_a = min(a[0], a[1])
    max_a = max(a[0], a[1])
    min_b = min(b[0], b[1])
    max_b = max(b[0], b[1])

    return max(0, min(max_a, max_b) - max(min_a, min_b))


def compute_overlap_coeff(src_layer, target_layer):
    
    nlev_src      = src_layer.shape[0]
    Nray          = src_layer.shape[1]
    nlev_target   = target_layer.shape[0]
    overlap_coeff = {}
    for n in range(Nray):
        N = Nray - n - 1
        N = n
        overlap_coeff[N] = {}
        for lev1 in range(nlev_target-1):
            overlap_coeff[N][lev1] = {}
            range1 = [target_layer[lev1], target_layer[lev1+1] ]
            total_overlap = 0
            for lev2 in range(nlev_src-1):
                range2        = [src_layer[lev2,n], src_layer[lev2+1,n]]
                overlap       = getOverlap(range1, range2)
                if overlap > 0:
                    overlap_coeff[N][lev1][lev2] = overlap
                total_overlap = total_overlap + overlap
            #    print(n,lev1,lev2,range1,range2,overlap, total_overlap)
            #exit()
            for lev2 in overlap_coeff[N][lev1]:
                overlap_coeff[N][lev1][lev2] = overlap_coeff[N][lev1][lev2] / total_overlap
                #if total_overlap < 240:
                #    overlap_coeff[n][lev1][lev2] = NaN
                #print(lev2,overlap_matrix[n][lev1][lev2])
            #print(lev1,overlap_coeff[n][lev1])
        #exit()
    return overlap_coeff


def format_levels(field, overlap_coeff):
    Nray   = field.shape[1]
    output = np.zeros((125,Nray))
    for n in range(Nray):
        for lev1 in range(125):
            #print(n,lev1,overlap_coeff[n][lev1])
            if bool(overlap_coeff[n][lev1]):
                value = 0
                for lev2 in overlap_coeff[n][lev1]:
                    coeff = overlap_coeff[n][lev1][lev2]
                    value = value + coeff * field[lev2,n]
                    #print('  ',value, coeff, cosp_dbze[lev2,n])
            else:
                value = NaN
            output[lev1][n] = value
    return output

#########################################################################################################################################################
#                                                                   INPUTS PARAMETRS                                                                    #
#########################################################################################################################################################

#xi=0
#yi=0
#xf=640
#yf=100

#xi=250
#xf=575
#yi=205
#yf=490

#ncfile_cs     = '/pampa/poitras/DATA/ORIGINAL/CLOUDSAT/NetCDF/CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00/20140105180611_40917_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.nc'
#ncfile_cosp   = '/home/poitras/SCRIPTS/COSPv2.0/driver3/data/P3/outputs/cosp_output_201401052000rz_2D.nc'
#ncfile_cospin = '/home/poitras/SCRIPTS/COSPv2.0/driver3/data/P3/inputs/cosp_input_201401052000rz.nc'
#ncfile_gem    = '/pampa/poitras/DATA/TREATED/GEM5/CORDEX/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/storm_019~/Samples_NetCDF_COSP/storm_019_201401/pm2014010100_20140105d.nc'





yi = int(sys.argv[1])
yf = int(sys.argv[2])
xi = int(sys.argv[3])
xf = int(sys.argv[4])
ncfile_cs        = sys.argv[5]
ncfile_cospout   = sys.argv[6]
ncfile_cospin    = sys.argv[7]
ncfile_gem       = sys.argv[8]
figure_directory = sys.argv[9]



if not os.path.exists(figure_directory                   ): os.makedirs(figure_directory                   )
if not os.path.exists(figure_directory + '/map'          ): os.makedirs(figure_directory + '/map'          )
if not os.path.exists(figure_directory + '/profil'       ): os.makedirs(figure_directory + '/profil'       )
if not os.path.exists(figure_directory + '/profil_gemlev'): os.makedirs(figure_directory + '/profil_gemlev')


# We are assuming the the file has this form: path/cosp_input_201401051800.nc
# Step 0 --> 01h00 ... step 23 --> 24h00: this is why we substract 1
filename_ext = os.path.basename(os.path.normpath(ncfile_cospin))
filename     = os.path.splitext(filename_ext)[0]
strdate      = filename[11:23]
t_gem        = int( strdate[8:10] ) - 1
nlev_cs  = 125
dz_cs    = 240




#########################################################################################################################################################
#                                                                                                                                                       #
#########################################################################################################################################################
border_11, lim_11 = generate_border('NAM-11m.nml','free','/chinook/poitras/shapefile/world_countries_boundary_file_world/world_countries_boundary_file_world_2002.shp',[0,0], [500,500])
grid_2p5          = read_gem_settings('Eastern_Canada_2p5.nml')
grid_11           = read_gem_settings('NAM-11m.nml')
box_11            = domainbox(grid_11 ,'free',grid_11,'free')
box_2p5           = domainbox(grid_2p5,'free',grid_11,'free')


resize_box_11_i = [xi, xf, xf, xi, xi];
resize_box_11_j = [yi, yi, yf, yf, yi];
#########################################################################################################################################################
#                                                                      READING DATA                                                                     #
#########################################################################################################################################################

# Data to extract
varlist_cs      = [ 'longitude', 'latitude', 'Height', 'DEM_elevation', 'Radar_Reflectivity' ]
varlist_cospout = [ 'clgrLidar532','clatlid','clcalipso']
varlist_cospin  = [ 'height', 'height_half', 'orography', 'latitude', 'longitude', 'mr_lsliq' ]
varlist_gem     = [ 'ZET','lon','lat' ]

# Creating empty dictionnary (they will store data + attributes)
data_cs       = {}; attributes_cs     = {};
data_gem      = {}; attributes_gem    = {};
data_cosp     = {}; attributes_cosp   = {};
# Extracting data
netcdf4_extract_fields_and_attributes(varlist_cs     , [ncfile_cs     ], data_cs  , attributes_cs  );
netcdf4_extract_fields_and_attributes(varlist_gem    , [ncfile_gem    ], data_gem , attributes_gem );
netcdf4_extract_fields_and_attributes(varlist_cospout, [ncfile_cospout], data_cosp, attributes_cosp);
netcdf4_extract_fields_and_attributes(varlist_cospin , [ncfile_cospin ], data_cosp, attributes_cosp);

# Extra operation on cloudsat (cs) data and attributes
data_cs      ['time'] = set_time_cloudsat(ncfile_cs)
attributes_cs['time'] = {'longname': 'Date: YYYYMMDDhhmmss.f'}
format_data_cs(data_cs, attributes_cs)

# Extracting  the number of vertical level
nlev_gem = data_cosp['height'].shape[0]


#########################################################################################################################################################
#                                                                     Extracting tracks                                                                 #
#########################################################################################################################################################
# Selecting cloudsat (cs) data located inside the domain
cs_i, cs_j            = latlon2indx          (data_cs['longitude'], data_cs['latitude'], grid_11, 'free', 'lonlat2index')
cs_coord              = construct_coord      (cs_i                , cs_j                                                ) 
resize_box_11_coord   = construct_coord      (resize_box_11_i     , resize_box_11_j                                     )
cs_inside_box_11_flag = point_inside_polygon (cs_coord            , resize_box_11_coord                                 )
cs_inside_box_11_data = select_points        (data_cs             , cs_inside_box_11_flag                               )
cs_inside_box_11_i, cs_inside_box_11_j   = latlon2indx(cs_inside_box_11_data['longitude'],cs_inside_box_11_data['latitude'],grid_11,'free','lonlat2index')


print(cs_inside_box_11_data['time'][0],cs_inside_box_11_data['time'][-1])

# ...
Nray      = len(cs_inside_box_11_data['latitude'])
roundi    = np.round(cs_inside_box_11_i).astype(int)
roundj    = np.round(cs_inside_box_11_j).astype(int)
roundflag = np.ones(Nray)

# Elimating the repeating coordinates
# We will actually keep them the have the same number of point that cloudsat has.
#for n in range (1,Nray):
#    if roundi[n] == roundi[n-1] and roundj[n] == roundj[n-1]:
#        roundflag[n] = 0
#roundi = roundi[roundflag == 1]
#roundj = roundj[roundflag == 1]

# Preparing field to extract ( 2D + altitude --> 1D tracks + altitude)
# lat lon are use to make sure eveything is ok
gem_lon =  data_gem['lon'] - 360
gem_lat =  data_gem['lat']
gem_ZET =  data_gem['ZET'][t_gem,:,:,:]
gem_zet =  np.zeros((71,Nray))


cosp_lon     = data_cosp['longitude'] - 360
cosp_lat     = data_cosp['latitude' ]
cosp_ORO     = data_cosp['orography']
cosp_HEIGHT  = data_cosp['height'   ]
cosp_CLGRLIDAR532    = data_cosp['clgrLidar532'   ]
cosp_CLATLID    = data_cosp['clatlid'   ]
cosp_CLCALIPSO    = data_cosp['clcalipso'   ]
cosp_clgrlidar532    = np.zeros((40,Nray))
cosp_clatlid    = np.zeros((40,Nray))
cosp_clcalispo    = np.zeros((40,Nray))
cosp_height  = np.zeros((71,Nray))
cosp_oro     = np.zeros((   Nray))
cosp_layer   = np.zeros((72,Nray))

cosp_mr_lsliq    = np.zeros((71,Nray)); cosp_MR_LSLIQ = data_cosp['mr_lsliq']


cs_lat           = cs_inside_box_11_data['latitude'     ]
cs_lon           = cs_inside_box_11_data['longitude'    ] 
cs_height        = cs_inside_box_11_data['Height'       ]
cs_DEM_elevation = cs_inside_box_11_data['DEM_elevation']
cs_height        = cs_height - np.min(np.min(cs_height)) 
#cs_layer         = np.zeros((126,Nray))


#plt.plot(cs_height[:,-1])
#plt.show()



# Extracting fields
for n in range(Nray):
    #print(n,Roundi[n],Roundj[n],ZET.shape,zet.shape)
    I_gem    = roundj[n] - 1
    J_gem    = roundi[n] - 1
    I_cosp   = roundi[n] - xi - 1 
    J_cosp   = roundj[n] - yi - 1
    I_cospin = J_cosp
    J_cospin = I_cosp
    #print('%5d (%3d %3d) [%11.6f %11.6f] [%11.6f %11.6f]' % (n, roundj[n], roundi[n], cs_lon[n], cs_lat[n], gem_lon [I_gem   ,J_gem   ], gem_lat [I_gem   ,J_gem   ]))
    #print('%5d (%3d %3d) [%11.6f %11.6f] [%11.6f %11.6f]' % (n, roundj[n], roundi[n], cs_lon[n], cs_lat[n], cosp_lon[I_cospin,J_cospin], cosp_lat[I_cospin,J_cospin]))    
    gem_zet     [:,    n] = gem_ZET     [:, I_gem   , J_gem   ] 
    

    cosp_clgrlidar532[:,    n]    = cosp_CLGRLIDAR532   [:, I_cosp  , J_cosp  ]
    cosp_clatlid[:,    n]         = cosp_CLATLID        [:, I_cosp  , J_cosp  ]
    cosp_clcalispo[:,    n]       = cosp_CLCALIPSO      [:, I_cosp  , J_cosp  ]

    cosp_layer  [0,    n] = cosp_ORO    [   I_cospin, J_cospin]
    cosp_layer  [1:72, n] = cosp_HEIGHT [:, I_cospin, J_cospin]



    #cs_height   [n,    :] = cs_height   [n,:] + cs_DEM_elevation[n]    
    #cs_layer    [1:126,n] = np.flip(cs_height   [n,:].T)
    #cs_layer    [0,    :] = cs_DEM_elevation 


#print(cs_height.shape)
#plt.plot(cs_DEM_elevation,'r-')
#plt.plot(cs_layer[1,:],'b--')
#plt.plot(cosp_layer  [0,   :],'b-')
#plt.plot(cosp_layer  [1,   :],'b--')

#target_layer  = np.arange(0,126*240,240)
#overlap_coeff = compute_overlap_coeff(np.flipud(cosp_layer), np.flip(target_layer))
#overlap_cs    = compute_overlap_coeff(np.flip(cs_layer), np.flip(target_layer))




#gem_zet_cslevel   = format_levels(gem_zet  , overlap_coeff)
print('grd')
#cosp_clgrlidar532_cslevel = format_levels(cosp_clgrlidar532, overlap_coeff)
print('atlid')
#cosp_clatlid_cslevel = format_levels(cosp_clatlid, overlap_coeff)
print('calipso')
#cosp_clcalispo_cslevel = format_levels(cosp_clcalispo, overlap_coeff)




#########################################################################################################################################################
#                                                                      CREATING FIGURES                                                                 #
#########################################################################################################################################################

#y = [0,5,10,15,20,25,30]

#nlev = 125
#dz   = 240

#ytick = (nlev-1) - (yticklabel*1000)/dz

#
y         = np.arange(0,11*1920,1920) 
ylabel    = y.astype(str)
ytick     = (40) - y/480 -0.5 

print(ytick)



# Orography for GEM/COSP
orography   = (nlev_cs-1) - cosp_layer[0,:]/dz_cs
orography_x = np.append(np.arange(Nray), [Nray-1 , 0       , 0           ]) 
orography_y = np.append(orography      , [nlev_cs, nlev_cs , orography[0]])


# Orography for CloudSat
#orography_cloudsat = (105-1) - cs_DEM_elevation/dz_cs
#orography_cloudsat_x = np.append(np.arange(Nray)   , [Nray-1 , 0   , 0                    ])
#orography_cloudsat_y = np.append(orography_cloudsat, [105    , 105 , orography_cloudsat[0]])



cmap = cm.get_cmap('jet'    , 9); clim = [ -40, 50]

plt.figure(1)
cmap = cm.get_cmap('jet'    , 10); clim = [ 0, 100]
data = cosp_clgrlidar532
data [np.isnan(data)] = -1000
fig  = plt.imshow(data,cmap,interpolation='none')
fig.set_clim(vmin=clim[0], vmax=clim[1]);
axes=plt.gca()
axes.set_aspect('auto'         , adjustable='box')
plt.colorbar( ticks=     range(0,110,10))
plt.title ('Ground Lidar 532: Total cloud cover')
plt.ylabel('Altitude [m]')
#plt.ylim([ytick_cs[0],ytick_cs[-1]])
plt.yticks(ytick, ylabel)
#plt.plot(orography_cloudsat_x,orography_cloudsat_y,'k-',linewidth=0.5)
#plt.fill(orography_cloudsat_x,orography_cloudsat_y,'w')
plt.show(block=False)
figure_name = figure_directory + '/profil/cloud_cover_total_'  + strdate + '_groundlidar532'
plt.savefig(figure_name,dpi=150,bbox_inches='tight')
print(figure_name)


plt.figure(2)
data = cosp_clatlid
fig = plt.imshow(data,cmap, interpolation='none')
fig.set_clim(vmin=clim[0], vmax=clim[1]);
axes=plt.gca()
#axes.invert_yaxis()
axes.set_aspect('auto'         , adjustable='box')
plt.colorbar(ticks = range(0,110,10))

plt.title('Atlid: Total cloud cover')
plt.ylabel('Altitude [m]')
plt.yticks(ytick,ylabel)
#plt.yticks(ytick_gem,ylabel)
#plt.ylim([0,1920])
figure_name = figure_directory + '/profil/cloud_cover_total_'  + strdate  + '_atlid'
plt.savefig(figure_name,dpi=150,bbox_inches='tight')
plt.show(block=False)
print(figure_name)


plt.figure(3)
data = cosp_clcalispo
#fig  = plt.imshow(data,cmap,interpolation='none')
fig  = plt.imshow(data,cmap,interpolation='none')
fig.set_clim(vmin=clim[0], vmax=clim[1]);
axes = plt.gca()
#axes.invert_yaxis()
axes.set_aspect('auto'         , adjustable='box')
plt.colorbar(ticks = range(0,110,10))
plt.title('Calispo: Total cloud cover')
plt.ylabel('Altitude [m]')
plt.yticks(ytick,ylabel)
#plt.ylim([ytick_gem[0],ytick_gem[-1]])
figure_name = figure_directory + '/profil/cloud_cover_total_'  + strdate  + '_calipso'
plt.savefig(figure_name,dpi=150,bbox_inches='tight')
plt.show(block=False)
print(figure_name)




#plt.plot(cs_inside_box_11_data['Navigation_land_sea_flag'],'r-')
plt.figure(4)
plot_domain(border_11, lim_11,  box_11, box_2p5)
cs_i[cs_i<-10] = NaN
plt.plot(cs_i,cs_j,'r-')
plt.plot(cs_inside_box_11_i,cs_inside_box_11_j,'b-')
#plt.plot(cs_inside_box_11_i-10,cs_inside_box_11_j,'b--')
#plt.plot(cs_inside_box_11_i+10,cs_inside_box_11_j,'b--')
plt.plot(resize_box_11_i,resize_box_11_j,'b-')
figure_name = figure_directory + '/map/' + strdate
plt.savefig(figure_name,dpi=150,bbox_inches='tight')
plt.show(block=False)
plt.show()
print(figure_name)

