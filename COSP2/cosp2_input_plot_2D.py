import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import os
from   shapely.geometry         import Point
from   shapely.geometry.polygon import Polygon
from   datetime                 import datetime
from   datetime                 import timedelta
from   pylab                    import cm

import sys;                     sys.path.append('/home/poitras/SCRIPTS/mes_modules_python')
from   netcdf4_extra            import netcdf4_extract_fields_and_attributes
from   netcdf4_extra            import ncdump
from   generate_border          import generate_border
from   grid_projection          import domainbox
from   grid_projection          import read_gem_settings
from   grid_projection          import convert_rotlatlon_to_cartesian as latlon2indx
from   colorbar_extra           import set_colorbar_extend as cbext
NaN = np.nan




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
    start_timedt = start_time + timedelta(milliseconds=dt)  
    time         = np.zeros(N)
    for n in range(N):
        time[n] = (start_timedt + timedelta(milliseconds=Profile_time[n]*1000)).strftime(format_out)
    return time;

def format_data_cs(data,attributes):
    for var in data:
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
        overlap_coeff[n] = {}
        for lev1 in range(nlev_target-1):
            overlap_coeff[n][lev1] = {}
            range1 = [target_layer[lev1], target_layer[lev1+1] ]
            total_overlap = 0
            for lev2 in range(nlev_src-1):
                range2        = [src_layer[lev2,n], src_layer[lev2+1,n]]
                overlap       = getOverlap(range1, range2)
                if overlap > 0:
                    overlap_coeff[n][lev1][lev2] = overlap
                total_overlap = total_overlap + overlap
            #    print(n,lev1,lev2,range1,range2,overlap, total_overlap)
            #exit()
            for lev2 in overlap_coeff[n][lev1]:
                overlap_coeff[n][lev1][lev2] = overlap_coeff[n][lev1][lev2] / total_overlap
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
                value     = 0
                coeff_tot = 0
                for lev2 in overlap_coeff[n][lev1]:
                    if not np.isnan(field[lev2,n]):
                        coeff     = overlap_coeff[n][lev1][lev2]
                        value     = value     + coeff * field[lev2,n]
                        coeff_tot = coeff_tot + coeff
                if coeff_tot >= 0.5:
                    value = value / coeff_tot
                else:
                    value = NaN
                    #print('  ',value, coeff, cosp_dbze[lev2,n])
            else:
                value = NaN
            output[lev1][n] = value
    return output


def extract_track_and_zformat(field, roundi, roundj, overlap_coeff):
    Nray   = len(roundi)
    FIELD  = np.zeros((71,Nray)); 

    for n in range(Nray):
        I_cosp       = roundi[n] - xi - 1
        J_cosp       = roundj[n] - yi - 1
        I_cospin     = J_cosp
        J_cospin     = I_cosp
        FIELD[:,n] = field[:, I_cospin, J_cospin]

    FIELD  = np.flipud(FIELD)
    output = format_levels(FIELD, overlap_coeff)
    return output



#########################################################################################################################################################
#                                                                   INPUTS PARAMETRS                                                                    #
#########################################################################################################################################################

#xi=250
#xf=575
#yi=205
#yf=490
#ncfile_cs     = '/pampa/poitras/DATA/ORIGINAL/CLOUDSAT/NetCDF/CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00/20140105180611_40917_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.nc'
#ncfile_cospin = '/home/poitras/SCRIPTS/COSPv2.0/driver3/data/P3/inputs/cosp_input_201401051900rz.nc'

yi = int(sys.argv[1])
yf = int(sys.argv[2])
xi = int(sys.argv[3])
xf = int(sys.argv[4])
ncfile_cs        = sys.argv[7]
ncfile_cospin    = sys.argv[8]
figure_directory = sys.argv[9]


if not os.path.exists(figure_directory            ): os.makedirs(figure_directory            )
if not os.path.exists(figure_directory + '/map'   ): os.makedirs(figure_directory + '/map'   )
if not os.path.exists(figure_directory + '/profil'): os.makedirs(figure_directory + '/profil')


filename_ext = os.path.basename(os.path.normpath(ncfile_cospin))
filename     = os.path.splitext(filename_ext)[0]

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

resize_border_11      = {}
resize_border_11['i'] = border_11['i'] - xi
resize_border_11['j'] = border_11['j'] - yi
resize_lim_11         = {}
resize_lim_11['xlim'] = [ lim_11['xlim'][0] - xi,  lim_11['xlim'][1] - xi ]
resize_lim_11['ylim'] = [ lim_11['ylim'][0] - yi,  lim_11['ylim'][1] - yi ]
#########################################################################################################################################################
#                                                                      READING DATA                                                                     #
#########################################################################################################################################################

# Data to extract
varlist_cs     = [ 'longitude', 'latitude', 'Height', 'DEM_elevation' ]
varlist_cospin = [ 'landmask', 'skt', 'sunlit' ] + [ 'height', 'height_half', 'orography', 'latitude', 'longitude']


# Creating empty dictionnary (they will store data + attributes)
data_cs       = {}; attributes_cs     = {};
data_cosp     = {}; attributes_cosp   = {};

# Extracting data
netcdf4_extract_fields_and_attributes(varlist_cs    , [ncfile_cs    ], data_cs  , attributes_cs  );
netcdf4_extract_fields_and_attributes(varlist_cospin, [ncfile_cospin], data_cosp, attributes_cosp);

# Extra operation on cloudsat (cs) data and attributes
data_cs      ['time'] = set_time_cloudsat(ncfile_cs)
attributes_cs['time'] = {'longname': 'Date: YYYYMMDDhhmmss.f'}
format_data_cs(data_cs, attributes_cs)


# Extracting  the number of vertical level
nlev_gem = data_cosp['T_abs'].shape[0]


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
#gem_lon =  data_gem['lon'] - 360
#gem_lat =  data_gem['lat']

cs_lat           = cs_inside_box_11_data['latitude'     ]
cs_lon           = cs_inside_box_11_data['longitude'    ]
cs_height        = cs_inside_box_11_data['Height'       ]
cs_DEM_elevation = cs_inside_box_11_data['DEM_elevation']

COSPIN_lon       = data_cosp['longitude'] - 360
COSPIN_lat       = data_cosp['latitude' ]
COSPIN_orography = data_cosp['orography']
COSPIN_sunlit    = data_cosp['sunlit'   ]
COSPIN_skt       = data_cosp['skt'      ]
COSPIN_landmask  = data_cosp['landmask' ]

cospin_orography = np.zeros((Nray))
cospin_sunlit    = np.zeros((Nray))
cospin_skt       = np.zeros((Nray))
cospin_landmask  = np.zeros((Nray))


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
    cospin_orography[n] = COSPIN_orography[I_cospin, J_cospin]
    cospin_sunlit   [n] = COSPIN_sunlit   [I_cospin, J_cospin]
    cospin_skt      [n] = COSPIN_skt      [I_cospin, J_cospin]
    cospin_landmask [n] = COSPIN_landmask [I_cospin, J_cospin]







#########################################################################################################################################################
#                                                                      CREATING FIGURES                                                                 #
#########################################################################################################################################################

resize_lim_11         = {}
resize_lim_11['xlim'] = [ 0,  len(COSPIN_orography[0,:]) ]
resize_lim_11['ylim'] = [ 0,  len(COSPIN_orography[:,0]) ]

resize_cs_inside_box_11_i = cs_inside_box_11_i - xi
resize_cs_inside_box_11_j = cs_inside_box_11_j - yi

for i in range(4):

    
    if   i == 0 : data = cospin_orography; DATA = COSPIN_orography; title = 'Orography'       ; var = 'orography'; unit = 'm'; clim = [0  , 1200]; ncol = 12
    elif i == 1 : data = cospin_sunlit   ; DATA = COSPIN_sunlit   ; title = 'Sunlit'          ; var = 'sunlit'   ; unit = '' ; clim = [0  ,    1]; ncol =  2
    elif i == 2 : data = cospin_skt      ; DATA = COSPIN_skt      ; title = 'Skin temperature'; var = 'skt'      ; unit = 'k'; clim = [220,  320]; ncol = 10
    elif i == 3 : data = cospin_landmask ; DATA = COSPIN_landmask ; title = 'Land mask'       ; var = 'landmask' ; unit = '' ; clim = [0  ,    1]; ncol =  2

    plt.figure(i+1)
    cmap = cm.get_cmap('jet', ncol);
    fig  = plt.imshow(DATA,cmap)
    plot_domain(resize_border_11, resize_lim_11,  box_11, box_2p5)
    cs_i[cs_i<-10] = NaN
    plt.plot(cs_i,cs_j,'r-')
    plt.plot(resize_cs_inside_box_11_i,resize_cs_inside_box_11_j,'w--')

    fig.set_clim(vmin=clim[0], vmax=clim[1]);
    axes = plt.gca()
    axes.invert_yaxis()
    axes.set_aspect('auto'         , adjustable='box')
    #plt.colorbar(ticks = range(clim[0],clim[1]+1,1),label=unit,extend=cbext(data,clim))
    plt.colorbar(label=unit,extend=cbext(data,clim))
    #plt.colorbar()
    plt.title(title)
    plt.gca().invert_yaxis()
    figure_name = figure_directory + '/map/' + filename + '_' + var
    plt.savefig(figure_name,dpi=150,bbox_inches='tight')
    #plt.show(block=False)
    print(figure_name)

    plt.figure(i+11)
    fig  = plt.plot(data,'b-')
    plt.title(title)
    plt.ylabel(unit)
    plt.xlim(0,Nray)
    figure_name = figure_directory + '/profil/' + filename + '_' + var
    plt.savefig(figure_name,dpi=150,bbox_inches='tight')
    #plt.show(block=False)
    print(figure_name)



#plt.show()

