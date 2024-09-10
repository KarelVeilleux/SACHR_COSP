#!/usr/bin/env python3

__author__     = "Vincent Poitras"
__credits__    = "..."
__maintainer__ = "Karel veilleux?"
__email__      = "vincent.poitras@ec.gc.ca"
__status__     = "Development"


def get_args(argv, script='default'):
    '''Parse, assign values to global variables and return arguments passed in'''

    import argparse
    #import inspect

    args = ['configuration', 'cdo', 'dataset', 'nomvar','month','year', 'script']

    help_nomvar        = '''The name of the variable (fst naming convention) that you want to process.\neg: PR, PR0, TT, TD, O1, N0, I5.'''
    help_configuration = '''Path to the configuration file'''
    help_cdo           = '''Path to cdo binary (executable file)'''
    help_year          = '''Year to process'''
    help_month         = '''Month to process'''
    help_dataset       = '''Dataset to process (ERA5, MERRA2)'''
    help_script        = '''The name of the script which will be launched. This argument is (normaly) only used by sanity_check.py during the lauching process'''

    script_requiered_args = {}
    script_requiered_args['main_CALIPSO_vs_GEMCOSP_01_compute_profile.py'] = ['configuration', 'year', 'month']
    script_requiered_args['sanity_check.py'                              ] = ['script']
    script_requiered_args['default'                                      ] = []

    if script not in script_requiered_args:
        print('[ERROR] In %s: %s not found in the listed scripts' % (__file__, script))
        print('[ERROR] exit')
        exit()

    requiered_arg = {}
    for arg in args:
        requiered_arg[arg] = True if arg in script_requiered_args[script] else False

    
    #pmargs_logger.debug('Function get_args(%s)', argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('-n'  , '--nomvar'       , type=str, dest='nomvar'       , help=help_nomvar        , required=requiered_arg['nomvar'       ])
    parser.add_argument('-c'  , '--configuration', type=str, dest='configuration', help=help_configuration , required=requiered_arg['configuration'])
    parser.add_argument('-cdo', '--cdo'          , type=str, dest='cdo'          , help=help_cdo           , required=requiered_arg['cdo'          ])
    parser.add_argument('-ds' , '--dataset'      , type=str, dest='dataset'      , help=help_dataset       , required=requiered_arg['dataset'      ])
    parser.add_argument('-m'  , '--month'        , type=int, dest='month'        , help=help_month         , required=requiered_arg['month'        ])
    parser.add_argument('-y'  , '--year'         , type=int, dest='year'         , help=help_year          , required=requiered_arg['year'         ])
    parser.add_argument('-s'  , '--script'       , type=str, dest='script'       , help=help_script        , required=requiered_arg['script'       ])

    # Array for all arguments passed to script
    args = parser.parse_args(argv[1:])
   
    #caller = str(inspect.getouterframes(inspect.currentframe(), 2)[1][1])    
    return args.__dict__





def generate_filepath(dir_dataset, subdir_dataset, subdir, filetype, year, period=None, statistic=None, accumulation_period=None):
    """ Generate a pathfile

        :param dir_dataset: root directory for a given dataset
        :type  dir_dataset: str
        :param subdir_dataset: subdirectory of a specific field of a given dataset (see db_variable). eg. A-PR0-0, P-PR-0, M2T1NXSLV-T2M, M2T1NXASM-T2M, snow_depth, etc 
        :type  subdir_dataset: str
        :param subdir: subdirectory eg. download, hourly, mean_and_extremums, etc
        :type  subdir: str
        :param filetype: filetype [fst/NetCDF,png]
        :type  filetype: str
        :param year: year
        :type  year: int
        :param statistic: type of statistics [mean/minimum/maximum]
        :type  statistic: str
        :param period: type of period  [annual/monthly]
        :type  period: str
        :param accumulation_period: accumulation period in hour
        :type  accumulation_period: int
    """

    if filetype not in ['fst', 'NetCDF', 'png']:
        print('[ERROR] filetype = %s, must be fst or NetCDF' % filetype)
        print('[ERROR] exit')
        exit()

    if period not in ['annual', 'monthly', None]:
        print('[ERROR] period = %s, must be annual or monthly' % period)
        print('[ERROR] exit')
        exit()

    if statistic not in ['mean', 'maximum', 'minimum', None]:
        print('[ERROR] statistic = %s, must be mean, maximum or minimum' % period)
        print('[ERROR] exit')
        exit()


    if period is None: STATISTICS_SUBDIRS = ''
    else             : STATISTICS_SUBDIRS = '/' + period + '_' + statistic + '/' + '%03d' % accumulation_period

    if   filetype == 'NetCDF': EXTENSION = '.nc'
    elif filetype == 'fst'   : EXTENSION = ''
    elif filetype == 'png'   : EXTENSION = '.png'

    if   filetype == 'NetCDF': DATAorFIGURES = 'data'
    elif filetype == 'fst'   : DATAorFIGURES = 'data'
    elif filetype == 'png'   : DATAorFIGURES = 'figures'

    filepath = dir_dataset + '/' + subdir_dataset + '/' + DATAorFIGURES + '/' + filetype + '/' + subdir + STATISTICS_SUBDIRS + '/' + str(year) + EXTENSION 

    return filepath






def check_file_completness(filename, expected_number_of_timestep, test_last_timestep = True):
    ''' Check file completness by (1) checking if we have the expected number of time step and (2) checking if data of last time step (only NaN or constant value).
    
        :param filename: path to the file to be read
        :type  filename: str
        :param expected_number_of_timestep: expected number of timestep for a single field in a file. eg monthly mean of 1980: 12, hourly of 1980: 366*24
        :type  expected_number_of_timestep: int
        :param test_last_timestep : perform the test on the last timestep when True (default value) [True/False]
        :type test_last_timestep  : bool
        :return: file completness status [True/False]
        :rtype : bool  
    '''
    # Concerning the test on the last timestep
    #   It is assumed that if the field is constant, it is because the processing was incomplete and last timestep is just filled with _FillValue
    #   xarray is supposed to convert the missing_value/_FillValue into NaN, but there is an issue with the dowloaded ERA5 data 
    #       eg for snow_depth sd:_FillValue = -32767s ; but the value actually used to fill the download is -32766 (why?) and is not recognized as NaN
    #       Consequently we check for a constant field instead of only checking for a NaN field.
    #   Caveat 1: It is not impossible to have a field with a genuine constant value, eg annual minimum of precipitation (false incomplete)
    #   Caveat 2: It is not impossible that a problem occured while filling the last time-step, in this case the field might not be "constant" (false complete)
    #   
   
    import os
    import fstd2nc
    import numpy  as np
    import xarray as xr
    
    latnames = [ 'lat', 'latitude' , 'rlat' ]
    lonnames = [ 'lon', 'longitude', 'rlon' ]

    if not os.path.exists(filename): return False
    try:

        # Read file
        if '.nc' in filename: xarray = xr.open_dataset(filename)
        else                : xarray = fstd2nc.Buffer(filename,opdict='dummy').to_xarray()
        
        # Check last timestep
        if test_last_timestep == True:
            for var in xarray.data_vars:
                for latname, lonname in zip(latnames, lonnames):
                    if lonname in xarray[var].dims and latname in xarray[var].dims:
                        varname = var
            if 'time' in xarray[varname].dims: 
                data_last_time_step = xarray.isel(time=-1)[varname].values
            else: 
                data_last_time_step = xarray [varname].values
            if np.isnan(data_last_time_step).all() == True or np.all(data_last_time_step == data_last_time_step[0,0]):
                return False

        # Check the number of timestep
        number_of_timestep_in_the_file = len(xarray.time)
        if not number_of_timestep_in_the_file == expected_number_of_timestep:
            return False
    
    except:
        return False
    
    return True


def compress_netcdf(filein, fileout, method='cdo'):
    ''' Compress NetCDF file 

        :param filein: path to the file to be compressed (input)
        :type  filein: str
        :param fileout: path to the compressed file (output)
        :type  fileout: str
        :param method: method used to compress the file [cdo,xarray]
        :type  method: str
    '''
    import subprocess

    # Note: xarray method has the advantage of being purely pythonic, but using cdo is significantly faster and requiere less memory
    try:
        if method == 'cdo':
            subprocess.run([cdo,'-O','-L','-f','nc4c', '-z', 'zip_1','copy', filein, fileout])
        elif method == 'xarray':
            xarray = xr.open_dataset(filein)
            encode = {list(xarray.data_vars)[0]: {'zlib': True, 'complevel': 1}}
            xarray.to_netcdf(fileout, encoding=encode)

    except:
        print('[ERROR] An error occured during the compression of %s' % fileout)
        print('[ERROR] Exit')
        exit()





def makemap(data, file_data, file_png, title=None,  vext=[None,None], cmap=None, showcoastlines=True, features=[{}],  showcolorbar=True, extend=None, showfigure=False, savefigure=True, closefigure=True, overwrite=True):
    import os 
    import fstd2nc
    import numpy             as np
    import xarray            as xr
    import cartopy.crs       as ccrs
    import cartopy.feature   as cfeature
    import matplotlib.cm     as cm
    import matplotlib.pyplot as plt

    #from pylab               import cm


    if  os.path.exists(file_png) and overwrite == False:
        return 

    # Figure resolution
    dpi = 150

    # Read file_data
    if '.nc' in file_data: ds = xr.open_dataset(file_data)
    else                 : ds = fstd2nc.Buffer(file_data).to_xarray()
    
    # Extract latitude and longitude
    latnames = [ 'lat', 'latitude'  ]
    lonnames = [ 'lon', 'longitude' ]
    varname = list(ds.data_vars)[0]
    for latname, lonname in zip(latnames, lonnames):
        if lonname in ds.coords and latname in ds.coords:
            break
    lat = ds[latname].values.T
    lon = ds[lonname].values.T

    # Cartopy coordinates reference system
    if 'rotated_pole' in list(ds.data_vars):
        CCRS = ccrs.RotatedPole(ds.rotated_pole.grid_north_pole_longitude,
                                ds.rotated_pole.grid_north_pole_latitude)
    else:
        CCRS = ccrs.PlateCarree()

    # Set vext (if not set)
    if vext[0] is None:
        vext[0] = np.nanmin(np.nanmin(data))
        vext[1] = np.nanmax(np.nanmax(data))

    # Set cmap (if not set)
    if cmap is None:
        cmap = cm.get_cmap('jet',10)

    # Creating figure
    fig = plt.figure()
    fig.set_dpi(dpi)    
    ax = plt.axes(projection=CCRS)

    try:    pc = ax.pcolormesh(lon, lat, data  , cmap=cmap,vmin=vext[0], vmax=vext[1], transform=ccrs.PlateCarree())
    except: pc = ax.pcolormesh(lon, lat, data.T, cmap=cmap,vmin=vext[0], vmax=vext[1], transform=ccrs.PlateCarree())
  
    # Domain limits
    if  len(lon.shape) == 1:
        xll, yll = CCRS.transform_point(lon[ 0],lat[ 0], ccrs.PlateCarree())
        xur, yur = CCRS.transform_point(lon[-1],lat[-1], ccrs.PlateCarree())
    else:
        xll, yll = CCRS.transform_point(lon[ 0,  0],lat[ 0,  0], ccrs.PlateCarree())
        xur, yur = CCRS.transform_point(lon[-1, -1],lat[-1, -1], ccrs.PlateCarree())
        print(xll,xur,yll,yur)
        #xll = -25; xur=20; yll=-25; yur=10
    ax.set_extent([xll, xur, yll, yur], crs=CCRS)

    # Coastlines
    if showcoastlines == True: ax.coastlines()

    # Cartopy features
    if features is not None:
        for f in features:
            category  = f['category' ] if 'category'  in f else 'cultural'
            name      = f['name'     ] if 'name'      in f else 'admin_1_states_provinces_lakes'
            scale     = f['scale'    ] if 'scale'     in f else '50m'
            edgecolor = f['edgecolor'] if 'edgecolor' in f else 'black'
            facecolor = f['facecolor'] if 'facecolor' in f else 'none'
            linewidth = f['linewidth'] if 'linewidth' in f else 0.3
            
            feature = cfeature.NaturalEarthFeature(category=category, name=name, scale=scale, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)
            ax.add_feature(feature)

    # Colorbar
    if showcolorbar == True:
        if extend is None:
            if    vext[0] == 0       : extend = 'max'
            else                     : extend = 'both'
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vext[0],vext[1]))
        sm._A = []
        dc = (vext[1]-vext[0])/cmap.N
        ticks = np.arange(vext[0],vext[1]+dc,1*dc)
        plt.colorbar(sm,ax=ax, extend=extend,ticks=ticks)


    # title
    if title is not None:
        plt.title(title, fontsize=8)

    # Save show and close figure
    if savefigure  == True:
        if not os.path.exists(os.path.dirname(file_png)):
            os.makedirs(os.path.dirname(file_png))
        plt.savefig(file_png,bbox_inches='tight', dpi=dpi)
    if showfigure  == True: plt.show()
    if closefigure == True: plt.close()
    
