import numpy             as np
import matplotlib.pyplot as plt
import netCDF4
import os
#from   shapely.geometry         import Point
#from   shapely.geometry.polygon import Polygon
from   datetime                 import datetime
from   datetime                 import timedelta
import math

import sys;                  sys.path.append('/home/veilleux/Projet/Projet_SACHR/analyses_and_figures/python_modules')
from   domain                import set_domain
from   geometry              import point_inside_polygon
from   netcdf4_extra         import netcdf4_extract_fields_and_attributes
from   netcdf4_extra         import ncdump
from   generate_border       import generate_border
from   grid_projection       import read_gem_settings
from   grid_projection       import convert_rotlatlon_to_cartesian as latlon2indx



def construct_coord(x,y):
    coord = []
    N     = len(x)
    for n in range(N):
        coord.append((x[n],y[n]))
    return coord

############################################################################################################################
def extract_satellite_track(ncfile, coord_domain,satellite='unspecified',timei='19000101000000', timef='22000101000000'):


    if coord_domain == 'global':
        coord_domain = [(-360,-90),  (360,-90), (360,90), (-360,90), (-360,-90)]
    #nc_id = netCDF4.Dataset(ncfile,'r');
    #nc_attrs, nc_dims, nc_vars = ncdump(nc_id, verb=True)

    if   satellite == 'cloudsat': lonname = 'longitude'; latname = 'latitude'; timename = 'time';
    elif satellite == 'calipso' : lonname = 'Longitude'; latname = 'Latitude'; timename = 'Profile_Time';
    else                        : lonname = 'longitude'; latname = 'latitude'; timename = 'time';
    # Extracting latitude and longitude
    varlist    = [ lonname, latname  ]
    data_in    = {};
    attributes = {};
    netcdf4_extract_fields_and_attributes(varlist, [ncfile], data_in, attributes);

    # Formating data (missing, offset, scale factor)
    if satellite == 'cloudsat':
        data_in   [timename] = set_time_cloudsat(ncfile)
        attributes[timename] = {'longname': 'Date: YYYYMMDDhhmmss.f'}
        format_data_cloudsat(data_in, attributes)

    if satellite == 'calipso':
        data_in[latname]  = data_in[latname]                    [:,1]  # 3 values: initial[0], central[1], final[2]
        data_in[lonname]  = data_in[lonname]                    [:,1]  # 3 values: initial[0], central[1], final[2]
        data_in[timename] = set_time_satellite(ncfile,'calipso')[:,1]  # 3 values: initial[0], central[1], final[2]


    # Constructing cloudsat coord:  (x1,x2...) (y1,y2,...) --> ( (x1,y1), (x2,y2), ... )
    coord_track = construct_coord (data_in[lonname],data_in[latname])

    # Setting a flag (1 = inside, 0 = outside)
    #   for the points inside the polygon
    #   for the points inside the time range (to implement)
    spatial_flag = point_inside_polygon(coord_track, coord_domain)

    Nray = int(np.sum(spatial_flag))

    data_out = {}
    data_out['longitude'] = np.empty(Nray, dtype=float)
    data_out['latitude' ] = np.empty(Nray, dtype=float)
    data_out['time'     ] = np.empty(Nray, dtype=object)
    data_out['index'    ] = np.empty(Nray, dtype=int)

    j = 0
    for i in range(len(spatial_flag)):
        if (spatial_flag[i] > 0):
            data_out['longitude'][j] = data_in[lonname ][i]
            data_out['latitude' ][j] = data_in[latname ][i]
            data_out['time'     ][j] = data_in[timename][i]
            data_out['index'    ][j] = i
            #print(i,spatial_flag[i], data_in[lonname][i], data_in[latname][i], data_in[timename][i])

            j = j + 1

    return data_out

##################################################################################################################################
def set_time_satellite(file,satellite):
    if satellite == 'cloudsat':
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

    elif satellite == 'calipso':
        format_prof  = '%Y-%m-%dT%H:%M:%S.%fZ'
        format_UTC   = '%Y%m%d%H%M%S.%f'
        format_out   = '%Y%m%d%H%M%S.%f'
        nc           = netCDF4.Dataset(file,'r');
        #Profile_time = nc['Profile_Time'][:]
        UTC_time     = nc['Profile_UTC_Time'][:]+20000000  #formatted as 'yymmdd.ffffffff','yy' last two digits of year, 'mm':month, 'dd',  day 'ffffffff'  fractional part of the day.

        I   = len(UTC_time[:,0])
        J   = len(UTC_time[0,:])
        UTC_time_str = np.empty((I,J),dtype=object)
        for i in range(I):
            for j in range(J):
                dec, utc_YMD = math.modf(UTC_time[i,j])
                dec, utc_H   = math.modf(24*dec)
                dec, utc_M   = math.modf(60*dec)
                dec, utc_S   = math.modf(60*dec)

                UTC_time_str[i,j]=str(int(utc_YMD)) + "{:02d}".format(int(utc_H)) + "{:02d}".format(int(utc_M)) + "{:02d}".format(int(utc_S)) + '.' + "{:.6f}".format(dec).split('.')[1]


        #start_time   = datetime.strptime(''.join(nc['vdata_metadata_vdf_Date_Time_at_Granule_Start'][:][0].astype('U13')),format_in)
        #end_time     = datetime.strptime(''.join(nc['vdata_metadata_vdf_Date_Time_at_Granule_End'][:][0].astype('U13')),format_in)
        #dt           = np.round((Profile_time - Profile_time[0][0]) * 1000 )[:,0]
        #time         = np.empty(N,dtype=object)
        #for n in range(N):
            #time[n] = (start_time + timedelta(milliseconds=dt[n])).strftime(format_out)

        return UTC_time_str

###################################################################################################################
def construct_profil_satellite(ncfile, track, varname,satellite='unspecified'):
    data_in    = {};
    attributes = {};
    netcdf4_extract_fields_and_attributes([varname], [ncfile], data_in, attributes);
    print(satellite, varname)
    if satellite == 'cloudsat' :
        format_data_cloudsat(data_in, attributes)
    elif satellite == 'calipso':
        if varname == 'Surface_Elevation_Statistics':
            data_in[varname] = data_in[varname][:,1] * 1000
        elif varname == 'Cloud_Layer_Fraction':
            data_in[varname] = data_in[varname]/30



    if len(data_in[varname].shape) == 2:
        data_out = data_in[varname][track['index'],0:-1].T
    else:
        data_out = data_in[varname][track['index']]
    
    return data_out

###################################################################################################################
def plot_borders_and_tracks(track, domain,  attribute={}):

    a = 'title'             ; title              = attribute[a] if a in attribute else ''
    a = 'figname'           ; figname            = attribute[a] if a in attribute else 'nofigure'
    a = 'track_marker'      ; track_marker       = attribute[a] if a in attribute else 'r-'
    a = 'border_marker'     ; border_marker      = attribute[a] if a in attribute else 'k-'
    a = 'r_domainbox_marker'; r_domainbox_marker = attribute[a] if a in attribute else 'b-'
    a = 'full_domain'       ; full_domain        = attribute[a] if a in attribute else True



    grid, xi, xf, yi, yf, d = set_domain(domain) 
    
    
    nfig = plt.gcf().number
    plt.figure(nfig+1)

    # Plot borders
    i            = 0
    shp_file    = {}
    shp_file[i] = '/chinook/poitras/shapefile/world_countries_boundary_file_world/world_countries_boundary_file_world_2002.shp'; i = i + 1
    shp_file[i] = '/chinook/poitras/shapefile/Great_Lakes/LakeSuperior/LakeSuperior.shp'                                       ; i = i + 1
    shp_file[i] = '/chinook/poitras/shapefile/Great_Lakes/LakeMichigan/LakeMichigan.shp'                                       ; i = i + 1
    shp_file[i] = '/chinook/poitras/shapefile/Great_Lakes/LakeHuron/LakeHuron.shp'                                             ; i = i + 1
    shp_file[i] = '/chinook/poitras/shapefile/Great_Lakes/LakeStClair/LakeStClair.shp'                                         ; i = i + 1
    shp_file[i] = '/chinook/poitras/shapefile/Great_Lakes/LakeErie/LakeErie.shp'                                               ; i = i + 1
    shp_file[i] = '/chinook/poitras/shapefile/Great_Lakes/LakeOntario/LakeOntario.shp'                                         ; i = i + 1
    for i in shp_file:
        border, lim = generate_border(grid,'free',shp_file[i], [0,0])
        plt.plot(border['i'], border['j'], border_marker, linewidth=0.75)


    # Plot track
    #grd = read_gem_settings(grid)
    i, j = latlon2indx(track ['longitude']  , track ['latitude']   , grid , 'free', 'lonlat2index')
    plt.plot(i, j, track_marker , linewidth=0.75);

    # Plot reduced domain box
    if full_domain == True:
        X = [xi, xf, xf, xi, xi]
        Y = [yi, yi, yf, yf, yi]
        plt.plot(X, Y, r_domainbox_marker, linewidth=0.75)


    # Set limit to show
    if full_domain == True:
        xmin=0; xmax = grid['ni'] - 2*grid['blend_H']
        ymin=0; ymax = grid['nj'] - 2*grid['blend_H']
    else:
        xmin = xi; xmax = xf
        ymin = yi; ymax = yf
    plt.axis('scaled')
    plt.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax);
    plt.xticks([]); plt.yticks([])

    plt.title(title, fontsize="x-large")

    if  not figname == 'nofigure':
        path = os.path.dirname(figname)
        if not os.path.exists(path): os.makedirs(path)
        plt.savefig(figname,dpi=150,bbox_inches='tight')
        if '.png' in figname: print(figname)
        else                : print(figname + '.png')

    return nfig


