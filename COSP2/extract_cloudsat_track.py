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

def format_data_clousat(data,attributes):
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






#yi = int(sys.argv[1])
#yf = int(sys.argv[2])
#xi = int(sys.argv[3])
#xf = int(sys.argv[4])
#ncfile_cs        = sys.argv[5]
#ncfile_cospout   = sys.argv[6]
#ncfile_cospin    = sys.argv[7]
#ncfile_gem       = sys.argv[8]
#figure_directory = sys.argv[9]



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
#                                                                      READING DATA                                                                     #
#########################################################################################################################################################
# Data to extract (time is extracted separately)
varlist_cloudsat = [ 'longitude', 'latitude']

# Creating empty dictionnary (they will store data + attributes)
data_cs       = {}; attributes_cs     = {};


# Extracting data
netcdf4_extract_fields_and_attributes(varlist_cloudsat, [ncfile_cloudsat], data_cloudsat, attributes_cloudsat);


# Extracting time from cloudsat
data_cloudsat      ['time'] = set_time_cloudsat(ncfile_cloudsat)
attributes_cloudsat['time'] = {'longname': 'Date: YYYYMMDDhhmmss.f'}

# Formating cloudsat data (missing, offset, scale factor)
format_data_cloudsat(data_cloudsat, attributes_cloudsat)



#########################################################################################################################################################
#                                                                     Extracting tracks                                                                 #
#########################################################################################################################################################
# Selecting cloudsat data located inside the domainsand inside the "resized box"
cloudsat_i, cloudsat_j  = latlon2indx     (data_cloudsat['longitude'], data_cloudsat['latitude'], grid_11, 'free', 'lonlat2index')
cloudsat_coord          = construct_coord (cloudsat_i                , cloudsat_j                                                ) 
resize_box_11_coord     = construct_coord (resize_box_11_i           , resize_box_11_j                                           )





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


cosp_lon     = data_cosp['longitude'] - 360
cosp_lat     = data_cosp['latitude' ]
cosp_ORO     = data_cosp['orography']
cosp_HEIGHT  = data_cosp['height'   ]
cosp_DBZE    = data_cosp['dbze94'   ]
cosp_dbze    = np.zeros((71,Nray))
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
    #gem_zet     [:,    n] = gem_ZET     [:, I_gem   , J_gem   ] 
    #cosp_dbze   [:,    n] = cosp_DBZE   [:, I_cosp  , J_cosp  ]
    #cosp_layer  [0,    n] = cosp_ORO    [   I_cospin, J_cospin]
    #cosp_layer  [1:72, n] = cosp_HEIGHT [:, I_cospin, J_cospin]




#########################################################################################################################################################
#                                                                      CREATING FIGURES                                                                 #
#########################################################################################################################################################

plt.figure(1)
plot_domain(border_11, lim_11,  box_11, box_2p5)
cs_i[cs_i<-10] = NaN
plt.plot(cs_i,cs_j,'r-')
plt.plot(cs_inside_box_11_i,cs_inside_box_11_j,'b-')
plt.plot(resize_box_11_i,resize_box_11_j,'b-')
figure_name = figure_directory + '/map/' + strdate
plt.savefig(figure_name,dpi=150,bbox_inches='tight')
#plt.show(block=False)
plt.show()
print(figure_name)

