import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import os
from   shapely.geometry         import Point
from   shapely.geometry.polygon import Polygon
from   datetime                 import datetime
from   datetime                 import timedelta
from   pylab                    import cm
from   matplotlib.ticker        import FormatStrFormatter
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
    #plt.plot(box1[0], box1[1], '-k', linewidth=0.75);
    #plt.plot(box2[0], box2[1], '-k', linewidth=0.75);
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
        #print(var)
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
                if coeff_tot >= 0.50:
                    value = value / coeff_tot
                else:
                    value = NaN
                    
            else:
                value = NaN
            output[lev1][n] = value
    return output


def extract_track_and_zformat(field, roundi, roundj, overlap_coeff):
    Nray   = len(roundi)
    FIELD  = np.zeros((nlev_gem,Nray)); 

    for n in range(Nray):
        I_cosp       = roundi[n] - xi - 1
        J_cosp       = roundj[n] - yi - 1
        I_cospin     = J_cosp
        J_cospin     = I_cosp
        FIELD[:,n] = field[:, I_cospin, J_cospin]

    FIELD              = np.flipud(FIELD)
    FIELD_interpolated = format_levels(FIELD, overlap_coeff)
    return FIELD, FIELD_interpolated

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
#ncfile_cs        = '/pampa/poitras/DATA/ORIGINAL/CLOUDSAT/NetCDF/CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00/20140105180611_40917_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.nc'
#ncfile_cospin    = '/pampa/poitras/DATA/TREATED/COSP2/zhipeng/2015051512_120m_p3_1cat_COSP_input.nc'
#figure_directory = '/pampa/poitras/figures/COSP2/zhipeng/INPUT'

yi = int(sys.argv[1])
yf = int(sys.argv[2])
xi = int(sys.argv[3])
xf = int(sys.argv[4])
ncfile_cs        = sys.argv[5]
ncfile_cospin    = sys.argv[6]
figure_directory = sys.argv[7]



if not os.path.exists(figure_directory                   ): os.makedirs(figure_directory                   )
if not os.path.exists(figure_directory + '/map'          ): os.makedirs(figure_directory + '/map'          )
if not os.path.exists(figure_directory + '/profil'       ): os.makedirs(figure_directory + '/profil'       )
if not os.path.exists(figure_directory + '/profil_gemlev'): os.makedirs(figure_directory + '/profil_gemlev')


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
varlist_cs      = [ 'longitude', 'latitude', 'Height', 'DEM_elevation' ]
varlist_cospin1 = [ 'tca','cca','q_v', 'pfull', 'phalf', 'T_abs', 'dtau_s', 'dtau_c', 'dem_s', 'dem_c' ]
varlist_cospin1 = varlist_cospin1 + [ 'mr_lsliq', 'mr_lsice', 'mr_lsrain', 'mr_lssnow', 'mr_lsgrpl' ]
varlist_cospin1 = varlist_cospin1 + [ 'mr_ccliq', 'mr_ccice', 'mr_ccrain', 'mr_ccsnow' ]
varlist_cospin2 = ['Reff']
varlist_cospin  = varlist_cospin1 + varlist_cospin2 + [ 'height', 'height_half', 'orography', 'latitude', 'longitude']


# Keep only data actually present in the file

nc_id = netCDF4.Dataset(ncfile_cospin,'r');
nc_attrs, nc_dims, nc_vars = ncdump(nc_id, verb=False)
var_to_keep = []
for var in varlist_cospin:
    if var in nc_vars:      
         var_to_keep = var_to_keep + [var]
varlist_cospin  = var_to_keep
var_to_keep = []
for var in varlist_cospin1:
    if var in nc_vars:
         var_to_keep = var_to_keep + [var]
varlist_cospin1  = var_to_keep


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
cs_lat           = cs_inside_box_11_data['latitude'     ]
cs_lon           = cs_inside_box_11_data['longitude'    ]
cs_height        = cs_inside_box_11_data['Height'       ]
cs_DEM_elevation = cs_inside_box_11_data['DEM_elevation']
cs_height        = cs_height - np.min(np.min(cs_height))

#COSPIN_lon       = data_cosp['longitude'] - 360
#COSPIN_lat       = data_cosp['latitude' ]
COSPIN_orography = data_cosp['orography']
COSPIN_height    = data_cosp['height'   ]
cospin_height    = np.zeros((nlev_gem  ,Nray))
cospin_oro       = np.zeros((           Nray))
cospin_layer     = np.zeros((nlev_gem+1,Nray))


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
    cospin_layer    [0,    n] = COSPIN_orography [   I_cospin, J_cospin]
    cospin_layer    [1:nlev_gem+1, n] = COSPIN_height    [:, I_cospin, J_cospin]
    


target_layer  = np.arange(0,(nlev_cs+1)*dz_cs,dz_cs)
overlap_coeff = compute_overlap_coeff(np.flipud(cospin_layer), np.flip(target_layer))


data_cosp['Reff_lsliq' ] = data_cosp['Reff'][0]
data_cosp['Reff_lsice' ] = data_cosp['Reff'][1]
data_cosp['Reff_lsrain'] = data_cosp['Reff'][2]
data_cosp['Reff_lssnow'] = data_cosp['Reff'][3]
data_cosp['Reff_lsgrpl'] = data_cosp['Reff'][8]
#data_cosp['Reff_ccliq' ] = data_cosp['Reff'][4]
#data_cosp['Reff_ccice' ] = data_cosp['Reff'][5]
#data_cosp['Reff_ccrain'] = data_cosp['Reff'][6]
#data_cosp['Reff_ccsnow'] = data_cosp['Reff'][7]


varlist_Reff = ['Reff_lsliq','Reff_lsice','Reff_lsrain','Reff_lssnow','Reff_lsgrpl','Reff_ccliq','Reff_ccice','Reff_ccrain','Reff_ccsnow']
varlist_Reff = ['Reff_lsliq','Reff_lsice','Reff_lsrain','Reff_lssnow','Reff_lsgrpl']
cospin  = {}
cospinz = {}
for var in varlist_cospin1 + ['height', 'height_half'] + varlist_Reff:
    data = data_cosp[var]

    if    'Reff_' in var:           data [data == 0 ] = NaN
    elif  'mr_'   in var:           data [data == 0 ] = NaN
    elif  'dtau_' in var:           data [data == 0 ] = NaN
    elif  'dem_'  in var:           data [data == 0 ] = NaN
    elif  var     in ['tca','cca']: data [data == 0 ] = NaN
    cospin[var], cospinz[var]  = extract_track_and_zformat( data, roundi, roundj, overlap_coeff)
    

#########################################################################################################################################################
#                                                                      CREATING FIGURES                                                                 #
#########################################################################################################################################################
y         = np.arange(0,24,4)
ylabel    = y.astype(str)
ytick_gem = (nlev_cs-1) - 1000*y/dz_cs
ytick_cs  = (105    -1) - 1000*y/dz_cs


# Orography for GEM/COSP
orography   = (nlev_cs-1) - cospin_layer[0,:]/dz_cs
orography_x = np.append(np.arange(Nray), [Nray-1 , 0       , 0           ])
orography_y = np.append(orography      , [nlev_cs, nlev_cs , orography[0]]) 


# Orography for CloudSat
orography_cloudsat = (105-1) - cs_DEM_elevation/dz_cs
orography_cloudsat_x = np.append(np.arange(Nray)   , [Nray-1 , 0   , 0                    ])
orography_cloudsat_y = np.append(orography_cloudsat, [105    , 105 , orography_cloudsat[0]])


resize_lim_11         = {}
resize_lim_11['xlim'] = [ 0,  len(COSPIN_orography[0,:]) ]
resize_lim_11['ylim'] = [ 0,  len(COSPIN_orography[:,0]) ]

resize_cs_inside_box_11_i = cs_inside_box_11_i - xi
resize_cs_inside_box_11_j = cs_inside_box_11_j - yi

levels = [0,15,30]
#levels = range(71)
subtitle = {}
for level in levels:
    height_min = np.nanmin(np.nanmin(COSPIN_height[level]))
    height_max = np.nanmax(np.nanmax(COSPIN_height[level]))
    subtitle[level] = 'GEM level %d [%.0f to %0.f m]' % (level+1, height_min, height_max)
    print(subtitle[level])


# Freezing point contour profil (on interpolate level)
T273z =           cospinz['T_abs'] > 273.15 
T273  = np.flipud(cospin ['T_abs'] > 273.15)
plt.figure(666)
cT273  = plt.contour(range(len(T273 [0,:])) , range(len(T273 [:,0])), T273 , levels=1, colors='black').allsegs[1]
cT273z = plt.contour(range(len(T273z[0,:])) , range(len(T273z[:,0])), T273z, levels=1, colors='black').allsegs[1]
#i=0

#fig  = plt.imshow(T273)
#for e0 in cT273.allsegs:
#    for e1 in e0:
#        if  i ==0: c = 'c--'
#        elif  i ==1: c = 'r--'
#        elif  i ==2: c = 'b--'
#        elif  i ==3: c = 'g--'

#        plt.plot(e1[:,0],e1[:,1],c)
#    i = i +1 

#cT273 = plt.contour(range(len(T273[0,:])) , range(len(T273[:,0])), T273, levels=1, colors='black').allsegs[1]
#for e in cT273: 
#    plt.plot(e[:,0],e[:,1],'ro')
#plt.plot(cT273[:,0],cT273[:,1],'r-')
#axes = plt.gca()
#axes.set_aspect('auto', adjustable='box')











i = 0
for var in cospinz:
    dataz = cospinz[var];
    data  = np.flipud(cospin [var])
    print(var, np.nanmin(np.nanmin(data)), np.nanmax(np.nanmax(data)) )
    cmapname = 'jet'
    if   var == 'tca'        : title = 'Cloud fraction [microphysics + subgrid]'          ; unit = ''  ; clim = [ 0, 1]                  ;  ncolor = 10;
    elif var == 'cca'        : title = 'Cloud fraction [subgrid]'                         ; unit = ''  ; clim = [ 0, 1]                  ;  ncolor = 10;
    elif var == 'q_v'        : title = 'Specific humidity'                                ; 
    elif var == 'pfull'      : title = 'Pressure [full level]'                            ; unit = 'Pa'; clim = [ 5000, 105000]          ;  ncolor = 10;
    elif var == 'phalf'      : title = 'Pressure [half level]'                            ; unit = 'Pa'; clim = [ 5000, 105000]          ;  ncolor = 10;
    elif var == 'T_abs'      : title = 'Air temperature'                                  ; unit = 'K' ; clim = [ 273.15-100, 273.15+100];  ncolor = 10; cmapname = 'seismic'
    elif var == 'dtau_s'     : title = 'Cloud optical depth at 0.67 micron [microphysics]'; unit = ''  ; clim = [ 0, 5    ]              ;  ncolor = 10;
    elif var == 'dtau_c'     : title = 'Cloud optical depth at 0.67 micron [subgrid]'     ; unit = ''  ; clim = [ 0, 5    ]              ;  ncolor = 10;
    elif var == 'dem_s'      : title = 'Cloud emissivity at 10.5 micron [microphysics]'   ; unit = ''  ; clim = [ 0, 1    ]              ;  ncolor = 10;
    elif var == 'dem_c'      : title = 'Cloud emissivity at 10.5 micron [subgrid]'        ; unit = ''  ; clim = [ 0, 1    ]              ;  ncolor = 10;
    elif var == 'height'     : title = 'Height (full level)'                              ; unit = 'm' ; clim = [ 0, 30000]              ;  ncolor = 10;
    elif var == 'height_half': title = 'Height (half level)'                              ; unit = 'm' ; clim = [ 0, 30000]              ;  ncolor = 10;
    elif var == 'mr_lsliq'   : title = 'Mixing ratio: Liquid-cloud [microphysics]'        ; 
    elif var == 'mr_lsice'   : title = 'Mixing ratio: Ice-cloud [microphysics]'           ; 
    elif var == 'mr_lsrain'  : title = 'Mixing ratio: Rain [microphysics]'                ; 
    elif var == 'mr_lssnow'  : title = 'Mixing ratio: Snow [microphysics]'                ; 
    elif var == 'mr_lsgrpl'  : title = 'Mixing ratio: Graupel [microphysics]'             ; 
    elif var == 'mr_ccliq'   : title = 'Mixing ratio: Liquid-cloud [subgrid]'             ; 
    elif var == 'mr_ccice'   : title = 'Mixing ratio: Ice-Cloud [subgrid]'                ; 
    elif var == 'mr_ccrain'  : title = 'Mixing ratio: Rain [subgrid]'                     ; 
    elif var == 'mr_ccsnow'  : title = 'Mixing ratio: Snow [subgrid]'                     ; 
    elif var == 'Reff_lsliq' : title = 'Effective radius: Liquid-cloud [microphysics]'    ; 
    elif var == 'Reff_lsice' : title = 'Effective radius: Ice-cloud [microphysics]'       ; 
    elif var == 'Reff_lsrain': title = 'Effective radius: Rain [microphysics]'            ; 
    elif var == 'Reff_lssnow': title = 'Effective radius: Snow [microphysics]'            ; 
    elif var == 'Reff_lsgrpl': title = 'Effective radius: Graupel [microphysics]'         ; 
    elif var == 'Reff_ccliq' : title = 'Effective radius: Liquid-cloud [subgrid]'         ; 
    elif var == 'Reff_ccice' : title = 'Effective radius: Ice-Cloud [subgrid]'            ; 
    elif var == 'Reff_ccrain': title = 'Effective radius: Rain [subgrid]'                 ; 
    elif var == 'Reff_ccsnow': title = 'Effective radius: Snow [subgrid]'                 ; 

    if    'Reff_' in var:  unit = '10$^{-6}$m'       ; clim = [ 0, 100] ;  ncolor = 10; dataz = dataz * 1e6    ; 
    elif  'mr_'   in var:  unit = 'log$_{10}$(kg/kg)'; clim = [ -10, -0];  ncolor = 10; dataz = np.log10(dataz);
    elif  'q_v'   in var:  unit = 'log$_{10}$(kg/kg)'; clim = [ -10, -0];  ncolor = 10; dataz = np.log10(dataz);

    if    'Reff_' in var:  data = data * 1e6    ; 
    elif  'mr_'   in var:  data = np.log10(data);
    elif  'q_v'   in var:  data = np.log10(data);

    dclim = (clim[1] - clim[0]) / ncolor



    # Profile on interpolate level
    plt.figure(i+1)
    cmap = cm.get_cmap(cmapname,ncolor);
    fig  = plt.imshow(dataz,cmap,zorder=0)
    for segment in cT273z: plt.plot(segment[:,0],segment[:,1],color="0.8",zorder=1)
    plt.fill(orography_x,orography_y,'w' ,zorder=2)
    plt.plot(orography_x,orography_y,'k-',zorder=3)
    fig.set_clim(vmin=clim[0], vmax=clim[1]);
    axes = plt.gca()
    axes.invert_yaxis()
    axes.set_aspect('auto', adjustable='box')
    if var == 'T_abs':
        fmt = FormatStrFormatter("%.2f")
        plt.colorbar(ticks = np.arange(clim[0], clim[1] + dclim, dclim),label=unit,extend=cbext(dataz,clim),format=fmt)
    else:
        plt.colorbar(ticks = np.arange(clim[0], clim[1] + dclim, dclim),label=unit,extend=cbext(dataz,clim))
    plt.title(title)
    plt.ylabel('Altitude [km]')
    plt.yticks(ytick_gem,ylabel)
    plt.ylim([ytick_gem[0],ytick_gem[-1]])
    figure_name = figure_directory + '/profil/'  + filename + '_' + var
    plt.savefig(figure_name,dpi=150,bbox_inches='tight')
    #plt.show(block=False)
    print(figure_name)



    # Profile on model levels
    plt.figure(i+1001)
    cmap = cm.get_cmap(cmapname,ncolor);
    fig  = plt.imshow(data,cmap,zorder=0)
    for segment in cT273: plt.plot(segment[:,0],segment[:,1],color="0.8",zorder=1)
    fig.set_clim(vmin=clim[0], vmax=clim[1]);
    axes = plt.gca()
    axes.invert_yaxis()
    axes.set_aspect('auto', adjustable='box')
    if var == 'T_abs':
        fmt = FormatStrFormatter("%.2f")
        plt.colorbar(ticks = np.arange(clim[0], clim[1] + dclim, dclim),label=unit,extend=cbext(data,clim),format=fmt)
    else:
        plt.colorbar(ticks = np.arange(clim[0], clim[1] + dclim, dclim),label=unit,extend=cbext(data,clim))
    plt.title(title)
    plt.ylabel('Model levels')
    #plt.ylim([ytick_gem[0],ytick_gem[-1]])
    figure_name = figure_directory + '/profil_gemlev/'  + filename + '_' + var
    plt.savefig(figure_name,dpi=150,bbox_inches='tight')
    #plt.show(block=False)
    print(figure_name)



    # Map
    j = 1
    for level in levels:
        plt.figure(i+1+100*j)
        DATA = data_cosp[var][level]
        if   'Reff_' in var: DATA = DATA * 1e6    ; DATA [DATA == 0 ] = NaN
        elif 'mr_'   in var: DATA = np.log10(DATA);
        elif 'q_v'   in var: DATA = np.log10(DATA);
        else               : DATA [DATA == 0 ] = NaN
        fig  = plt.imshow(DATA,cmap)
        plot_domain(resize_border_11, resize_lim_11,  box_11, box_2p5)
        cs_i[cs_i<-10] = NaN
        plt.plot(resize_cs_inside_box_11_i,resize_cs_inside_box_11_j,'k--')
        fig.set_clim(vmin=clim[0], vmax=clim[1]);
        axes = plt.gca()
        axes.set_aspect('auto', adjustable='box')
        if var == 'T_abs':
            fmt = FormatStrFormatter("%.2f")
            plt.colorbar(ticks = np.arange(clim[0], clim[1] + dclim, dclim),label=unit,extend=cbext(data,clim),format=fmt)
        else:
            plt.colorbar(ticks = np.arange(clim[0], clim[1] + dclim, dclim),label=unit,extend=cbext(data,clim))

        plt.title(title + '\n' + subtitle[level])
        figure_name = figure_directory + '/map/' + filename + '_' + var + '_' + str(level)
        plt.savefig(figure_name,dpi=150,bbox_inches='tight')
        #plt.show(block=False)
        print(figure_name)
        j = j + 1
    i = i + 1


#plt.show()

