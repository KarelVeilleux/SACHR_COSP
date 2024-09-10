import numpy             as np
import matplotlib.pyplot as plt
import netCDF4
import os
from   shapely.geometry         import Point
from   shapely.geometry.polygon import Polygon
from   datetime                 import datetime
from   datetime                 import timedelta
from   pylab                    import cm
import math

import sys;                  sys.path.append('/home/poitras/SCRIPTS/mes_modules_python')
from   grid_projection       import read_gem_settings
from   grid_projection       import convert_rotlatlon_to_cartesian as latlon2indx
from   netcdf4_extra         import netcdf4_extract_fields_and_attributes
from   netcdf4_extra         import ncdump
from   generate_border       import generate_border
from   colorbar_extra        import set_colorbar_extend as cbext

NaN = np.nan
###########################################################################################


def set_domain(domain):
        if   domain == 'box11'            : grid = read_gem_settings('NAM-11m.nml');  xi=250; xf=575; yi=205; yf=490; d=0.1;
        elif domain == 'NAM11'            : grid = read_gem_settings('NAM-11m.nml');  xi=  0; xf=655; yi=  0; yf=655; d=0.1;
        elif domain == 'NAM-11m'          : grid = read_gem_settings('NAM-11m.nml');  xi=  0; xf=655; yi=  0; yf=655; d=0.1;
        elif domain == 'great_lakes_11'   : grid = read_gem_settings('NAM-11m.nml');  xi=315; xf=430; yi=262; yf=338; d=0.1; #Great Lakes
        elif domain == 'sonora_desert_11' : grid = read_gem_settings('NAM-11m.nml');  xi=130; xf=185; yi=140; yf=230; d=0.1; #Sonara desert
        elif domain == 'bermuda_azores_11': grid = read_gem_settings('NAM-11m.nml');  xi=530; xf=615; yi=145; yf=230; d=0.1; #Bermuda-Azores High
        elif domain == 'hudson_bay_11'    : grid = read_gem_settings('NAM-11m.nml');  xi=290; xf=410; yi=355; yf=480; d=0.1; #Hudson bay
        elif domain == 'pacific_sw_11'    : grid = read_gem_settings('NAM-11m.nml');  xi= 40; xf=100; yi=100; yf=200; d=0.1; #Pacific-SW
        elif domain == 'mexico_south_11'  : grid = read_gem_settings('NAM-11m.nml');  xi=230; xf=400; yi= 25; yf= 90; d=0.1; #Mexico-south
        elif domain == 'bc_coast_11'      : grid = read_gem_settings('NAM-11m.nml');  xi= 60; xf=190; yi=355; yf=480; d=0.1; #BC
        elif domain == 'ECAN2p5'          : print('Not implemented yet'); exit()
        else:                               grid = read_gem_settings(domain + '.nml'); xi = 0; yi = 0; di=0.1
        return grid, xi, xf, yi, yf, d



def generate_domain_coord(domain):
    
    grid, xi, xf, yi, yf, d = set_domain(domain) 
    seg1_i = np.arange(xi, xf+d, d); seg1_j = seg1_i * 0 + yi
    seg2_j = np.arange(yi, yf+d, d); seg2_i = seg2_j * 0 + xf
    seg3_i = np.arange(xf, xi-d,-d); seg3_j = seg3_i * 0 + yf
    seg4_j = np.arange(yf, yi-d,-d); seg4_i = seg4_j * 0 + xi
    i = np.concatenate((seg1_i, seg2_i, seg3_i, seg4_i))
    j = np.concatenate((seg1_j, seg2_j, seg3_j, seg4_j))
    lon, lat = latlon2indx(i  , j  , grid , 'free', 'index2lonlat')

    coord = construct_coord(lon,lat)

    return coord 


def generate_track_indices(track, domain):
       
    
    


    #if k==5 : xi= 60; xf=190; yi=355; yf=480; d=0.1; #BC
        grid, xi, xf, yi, yf, d = set_domain(domain)
        #if domain == 'box11':
        #    xi = 250 
        #    yi = 205
        #    grid  = read_gem_settings('NAM-11m.nml')
        #else:
        #    xi = 0
        #    yi = 0
        #    grid  = read_gem_settings(domain + '.nml')

        track_i, track_j = latlon2indx(track['longitude'], track['latitude'], grid, 'free', 'lonlat2index')
        track_roundi     = np.round(track_i).astype(int)
        track_roundj     = np.round(track_j).astype(int)

        Nray = len(track_i)
        indx = {}
        indx['i'] = np.empty(Nray, dtype=int)
        indx['j'] = np.empty(Nray, dtype=int)
        for n in range(Nray):
            indx['j'][n] = track_roundi[n] - xi - 1
            indx['i'][n] = track_roundj[n] - yi - 1
       
        return indx

#############################################################################################
#                                             Figures                                       #
#############################################################################################
def plot_profil(fields, varname, attribute={}):

    a = 'figname' ; figname  = attribute[a] if a in attribute else 'nofigure'
    a = 'vstruct' ; vstruct  = attribute[a] if a in attribute else 'calipso'
    a = 'masktype'; masktype = attribute[a] if a in attribute else 'none'

    # Extract data + data dimensions

    data = fields[varname]
    Nray = len(data[0,:])
    Nlev = len(data[:,0])
    
    # Construct ylabels
    if   vstruct == 'cloudsat': yi=0; dy=   5; ny= 4; dlev = 240; scalef = 1000; offset = 0.0; ylabel = 'Altitude [km]'
    elif vstruct == 'calipso' : yi=0; dy=1920; ny=10; dlev =  60; scalef =    1; offset = 0.5; ylabel = 'Altitude [m]'
    #elif vstruct == 'calipso' : yi=0; dy=1920; ny=10; dlev = 60; scalef =    1; offset = 0.5; ylabel = 'Altitude [m]'
    y          = np.arange(yi,(ny+1)*dy,dy)
    yticklabel = y.astype(str)
    ytick      = (Nlev-1) - scalef*y/dlev + offset

    #dlev = 240
    #y         = np.arange(0,6*4,4)
    #ylabel    = y.astype(str)
    #ytick  = (Nlev-1) - 1000*y/dlev

    
    
    # Construct the topography (if present in the input fields)
    if 'orography' in fields:
        orography   = (Nlev-1) - fields['orography']/dlev
        orography_x = np.append(np.arange(Nray), [Nray-1, 0    , 0           ])
        orography_y = np.append(orography      , [Nlev  , Nlev , orography[0]])
    else:
        orography_x = NaN
        orography_y = NaN

    if   varname == 'radar_reflectivity': cmap = cm.get_cmap('jet',  9); clim = [ -40,  50]; cticks =     range(-50, 60  ,10  )
    elif varname == 'total_cloud_cover' : cmap = cm.get_cmap('jet', 10); clim = [   0,   1]; cticks = np.arange(  0,  1.1, 0.1)
    else                                : cmap = cm.get_cmap('jet', 10); clim = [   0,  10]; cticks = np.arange(  0, 10  , 1  )


    title = attribute['title'] if 'title' in attribute  else ''


    # Figure
    plt.figure()
    fig  = plt.imshow(data,cmap,interpolation='none')
    #fig  = plt.imshow(data,cmap)
    plt.plot(orography_x,orography_y,'k-',linewidth=0.5)
    plt.fill(orography_x,orography_y,'w')

    fig.set_clim(vmin=clim[0], vmax=clim[1]);
    axes=plt.gca()
    axes.set_aspect('auto', adjustable='box')
    plt.colorbar(ticks = cticks)
    plt.ylim([ytick[0],ytick[-1]])
    plt.yticks(ytick, yticklabel)
    plt.title (title)
    plt.ylabel(ylabel)

    if  not figname == 'nofigure':
        plt.savefig(figname,dpi=150,bbox_inches='tight')
        print(figname + '.png')



def get_vertical_labeling(vstruct,Nlev):
    if   vstruct == 'cloudsat': zi=0; dz=   5; nz= 4; dlev = 240; scalef = 1000; offset = 0.0; zlabel = 'Altitude [km]'
    elif vstruct == 'calipso' : zi=0; dz=1920; nz=10; dlev =  60; scalef =    1; offset = 0.5; zlabel = 'Altitude [m]'
    z          = np.arange(zi,(nz+1)*dz,dz)
    zticklabel = z.astype(str)
    ztick      = (Nlev-1) - scalef*z/dlev + offset
    return ztick, zticklabel, zlabel, z


def plot_frequency_intensity_profil(data, varname, threshold, attribute):
    a = 'figname' ; figname  = attribute[a] if a in attribute else 'nofigure'
    a = 'vstruct' ; vstruct  = attribute[a] if a in attribute else 'calipso'
    a = 'masktype'; masktype = attribute[a] if a in attribute else 'show_sealand'


    if   masktype == 'show_sea' : subtitle = '\n[Over water]'
    elif masktype == 'show_land': subtitle = '\n[Over land]'
    else                        : subtitle = '\n'

    if threshold == 0:
        t1 = 'Number of pixels' + subtitle
        t2 = 'Mean cloud cover (all-sky)' + subtitle
    elif threshold == 1e-6:
        t1 = 'Number of cloudy pixels' + subtitle
        t2 = 'Mean cloud cover (cloudy-sky)' + subtitle
    elif threshold == 0.05:
        t1 = 'Number of cloudy pixels (cc>0.05)' + subtitle
        t2 = 'Mean cloud cover (cloudy-sky) (cc>0.05)' + subtitle
    else: 
        t1 = 'Number of cloudy pixels' + subtitle
        t2 = 'Mean cloud cover (cloudy-sky)' + subtitle

    


    if  not figname == 'nofigure':
        path = os.path.dirname(figname)   
        print('xxxx',path)
        if not os.path.exists(path):
            os.makedirs(path)

    # Extract Nlev and Nray from the 1st dataset
    first_dataset =  list(data.keys())[0]
    Nlev = len(data[first_dataset][varname][:,0])
    Nray = len(data[first_dataset][varname][0,:])
    
    nfig = plt.gcf().number

    for dataset in data:
        if   dataset == 'calipso': marker = 'k-'
        elif dataset == 'cospout': marker = 'r-'
        elif dataset == 'gem'    : marker = 'b-'

        field  = data[dataset][varname][::-1]
        flag   = field >= threshold
        notNaN = ~np.isnan(field)
        #Nmax   = len(field[0,:])
        Nmax   = np.sum(notNaN ,axis=1)
        N      = np.sum(flag   ,axis=1)
        I      = np.nansum(field * flag,axis=1) / N
        if   dataset == 'calipso': x = range(30,Nlev*60,60)
        else                     : x = range(30,Nlev*60,60)
        
        plt.figure(nfig+1) 
        plt.plot(N/Nmax*100,x,marker)

        plt.figure(nfig+2)
        plt.plot(I,x,marker)

    # Figure 1
    plt.figure(nfig+1)
    #plt.title('Events detected (cloud cover >= %.2f)' % threshold)
    plt.title(t1)
    plt.xlabel('% of pixels')
    ztick, zticklabel, zlabel, z =  get_vertical_labeling(vstruct,Nlev)
    plt.yticks(z)
    plt.ylabel(zlabel)
    plt.ylim([z[0],z[-1]])
    plt.xlim([-0,100])
    plt.axes().set_aspect(0.15*100/1920)
    print('debug')
    if  not figname == 'nofigure':
        fn = figname.replace('xxxYYY','frequency')
        plt.savefig(fn,dpi=150,bbox_inches='tight')
        print(fn + '.png')
    
    
    
    plt.figure(nfig+2)
    #plt.title('Events mean value (cloud cover >= %.2f)' % threshold)
    plt.title(t2)
    plt.xlabel('Cloud cover')
    ztick, zticklabel, zlabel, z =  get_vertical_labeling(vstruct,Nlev)
    plt.yticks(z)
    plt.ylabel(zlabel)
    plt.ylim([z[0],z[-1]])
    plt.xlim([0,1.01])
    plt.axes().set_aspect(0.15*1/1920)
    if  not figname == 'nofigure':
        fn = figname.replace('xxxYYY','intensity')
        plt.savefig(fn,dpi=150,bbox_inches='tight')
        print(fn + '.png')


#############################################################################################
def plot_map(data,domain,varname,track={'i':NaN, 'j':NaN}, attribute={}):
    
    if  len(data.shape)==1:  return()
    

    a = 'title'         ; title          = attribute[a] if a in attribute else ''
    a = 'cbar_labelsize'; cbar_labelsize = attribute[a] if a in attribute else 15
    a = 'figname'       ; figname        = attribute[a] if a in attribute else 'nofigure'
    a = 'datatype'      ; datatype       = attribute[a] if a in attribute else 'value'

    if   'cloud_fraction'       in varname and datatype == 'value'     : cmap = cm.get_cmap('viridis', 20); clim = [  0.00, 1.00]
    elif 'cloud_fraction'       in varname and datatype == 'difference': cmap = cm.get_cmap('seismic', 11); clim = [ -1.10, 1.10]
    elif 'cloud_water_path'     in varname and datatype == 'value'     : cmap = cm.get_cmap('jet'    , 20); clim = [  0, 2] 
    elif 'cloud_water_path'     in varname and datatype == 'difference': cmap = cm.get_cmap('seismic', 21); clim = [  -1.05, 1.05]
    elif 'optical_thickness'    in varname and datatype == 'value'     : cmap = cm.get_cmap('jet'    , 20); clim = [  0, 100]
    elif 'optical_thickness'    in varname and datatype == 'difference': cmap = cm.get_cmap('seismic', 21); clim = [  -52.25, 52.25]
    elif 'cloud_top_pressure'   in varname and datatype == 'value'     : cmap = cm.get_cmap('jet'    , 20); clim = [  0, 1000]
    elif 'cloud_top_pressure'   in varname and datatype == 'difference': cmap = cm.get_cmap('seismic', 21); clim = [  -210, 210]
    elif 'particle_size_ice'    in varname and datatype == 'value'     : cmap = cm.get_cmap('jet'    , 10); clim = [  0, 50]
    elif 'particle_size_ice'    in varname and datatype == 'difference': cmap = cm.get_cmap('seismic', 21); clim = [  -21, 21]
    elif 'particle_size_liquid' in varname and datatype == 'value'     : cmap = cm.get_cmap('jet'    , 10); clim = [  0, 25]
    elif 'particle_size_liquid' in varname and datatype == 'difference': cmap = cm.get_cmap('seismic', 21); clim = [  -10.5, 10.5]
    else                                                               : cmap = cm.get_cmap('viridis', 20); clim = [  -1.00, 1.00]      

    if   datatype == 'value'      : facecolor = 'xkcd:light grey'
    elif datatype == 'difference' : facecolor = 'xkcd:light grey'

    


    # Generate border
    if   domain == 'box11'  : grid  = 'NAM-11m.nml'; xi=250; xf=575; yi=205; yf=490;
    elif domain == 'NAM11'  : grid  = 'NAM-11m.nml'; xi=  1; xf=655; yi=  1; yf=655;
    elif domain == 'ECAN2p5': print('Not implemented yet')
    shp_file    = '/chinook/poitras/shapefile/world_countries_boundary_file_world/world_countries_boundary_file_world_2002.shp'
    
    border, lim = generate_border(grid,'free',shp_file, [-xi-1,-yi-1])

    # Creating the figure
    plt.figure()
    f = plt.imshow(data,cmap,interpolation='none')
    f.set_clim(vmin=clim[0], vmax=clim[1]);

    # Colorbar
    cbar = plt.colorbar(extend=cbext(data,clim))
    cbar.ax.tick_params(labelsize=cbar_labelsize)

    axes = plt.gca()
    axes.invert_yaxis()

    axes.set_facecolor(facecolor)
    plt.axis('scaled')
    plt.axis(xmin=0, xmax=xf-xi, ymin=0, ymax=yf-yi);
    plt.xticks([]); plt.yticks([])
    
   
    plt.title(title, fontsize="x-large")

    # Plot boundaries and track
    plt.plot(track ['j'], track ['i'], '--k', linewidth=0.75);
    plt.plot(border['i'], border['j'], '-k' , linewidth=0.75);
    
    # Plot Great Lakes
    shp_file = '/chinook/poitras/shapefile/Great_Lakes/LakeSuperior/LakeSuperior.shp'; border, lim = generate_border(grid,'free',shp_file, [-xi-1,-yi-1]); plt.plot(border['i'], border['j'], '-k',linewidth=0.75);
    shp_file = '/chinook/poitras/shapefile/Great_Lakes/LakeMichigan/LakeMichigan.shp'; border, lim = generate_border(grid,'free',shp_file, [-xi-1,-yi-1]); plt.plot(border['i'], border['j'], '-k',linewidth=0.75);
    shp_file = '/chinook/poitras/shapefile/Great_Lakes/LakeHuron/LakeHuron.shp'      ; border, lim = generate_border(grid,'free',shp_file, [-xi-1,-yi-1]); plt.plot(border['i'], border['j'], '-k',linewidth=0.75);
    shp_file = '/chinook/poitras/shapefile/Great_Lakes/LakeStClair/LakeStClair.shp'  ; border, lim = generate_border(grid,'free',shp_file, [-xi-1,-yi-1]); plt.plot(border['i'], border['j'], '-k',linewidth=0.75);
    shp_file = '/chinook/poitras/shapefile/Great_Lakes/LakeErie/LakeErie.shp'        ; border, lim = generate_border(grid,'free',shp_file, [-xi-1,-yi-1]); plt.plot(border['i'], border['j'], '-k',linewidth=0.75);
    shp_file = '/chinook/poitras/shapefile/Great_Lakes/LakeOntario/LakeOntario.shp'  ; border, lim = generate_border(grid,'free',shp_file, [-xi-1,-yi-1]); plt.plot(border['i'], border['j'], '-k',linewidth=0.75);

    if  not figname == 'nofigure':
        plt.savefig(figname,dpi=150,bbox_inches='tight')
        print(figname + '.png')




def plot_borders_and_tracks(track, domain,  attribute={}):

    a = 'title'             ; title              = attribute[a] if a in attribute else ''
    a = 'figname'           ; figname            = attribute[a] if a in attribute else 'nofigure'
    a = 'track_marker'      ; track_marker       = attribute[a] if a in attribute else 'r-'
    a = 'border_marker'     ; border_marker      = attribute[a] if a in attribute else 'k-'
    a = 'r_domainbox_marker'; r_domainbox_marker = attribute[a] if a in attribute else 'b-'
    a = 'full_domain'       ; full_domain        = attribute[a] if a in attribute else True
    
    # Generate border
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
        grid  = 'NAM-11m.nml'; 
        border, lim = generate_border(grid,'free',shp_file[i], [0,0])
        plt.plot(border['i'], border['j'], border_marker, linewidth=0.75)

   
    # Plot track
    grd = read_gem_settings(grid)
    i, j = latlon2indx(track ['longitude']  , track ['latitude']   , grd , 'free', 'lonlat2index')
    plt.plot(i, j, track_marker , linewidth=0.75);
    
    # Plot reduced domain box     
    if full_domain == True: 
        X = [xi, xf, xf, xi, xi]
        Y = [yi, yi, yf, yf, yi]
        plt.plot(X, Y, r_domainbox_marker, linewidth=0.75)
 

    # Set limit to show
    if full_domain == True:
        xmin=0; xmax = grd['ni'] - 2*grd['blend_H']
        ymin=0; ymax = grd['nj'] - 2*grd['blend_H']
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
        print(figname + '.png')

    return nfig

#############################################################################################
#                                          COSP/GEM                                         #
#############################################################################################
def construct_profil_2Ddata(ncfile, indx, varname, t=NaN):
    # Expected dimension structure: ({t},{z},x,y)
    # t (timestep) and z (vertical level) may be absent
    # If several timestep are present, the input variable t (the time step number whitin the
    #   file must be specified
    
    data_in    = {};
    attributes = {};
    netcdf4_extract_fields_and_attributes([varname], [ncfile], data_in, attributes)

    if np.isnan(t):
        data = data_in[varname]
    else:
        data = data_in[varname][t]
  
    if varname == 'clcalipso'   : data    = data   * 0.01  # %        --> 0-1
  
    Nray = len(indx['i'])
    Ndim = len(data.shape)

    if Ndim == 3:
        Nlev     = len(data[:,0,0])
        data_out = np.empty((Nlev,Nray))
    elif Ndim == 2:
        data_out = np.empty((Nray))

    for n in range(Nray):
        if Ndim == 3:
            data_out[:,n] = data[:, indx['i'][n], indx['j'][n]]
        elif Ndim == 2:
            data_out[n]   = data[   indx['i'][n], indx['j'][n]]
    
    return(data_out)




def construct_model_layer(ncfile, indx):

    data_in    = {};
    attributes = {};
    netcdf4_extract_fields_and_attributes(['orography', 'height'], [ncfile], data_in, attributes)

    Nray  = len(indx['i'])
    Nlev  = len(data_in['height'][:,0,0])
    layer = np.empty((Nlev+1, Nray))

    for n in range(Nray):

        layer[0,      n] = data_in['orography'][   indx['i'][n], indx['j'][n]]
        layer[1:Nlev+1, n] = data_in['height']   [:, indx['i'][n], indx['j'][n]]

    return layer;



def mask_orography(data,layer,varname,mask_value=NaN):
    orography = data['orography']
    var       = data[varname]
    Nlev      = len (var[:,0])
    Nray      = len (var[0,:])

    for n in range(Nray):
        lev = 0
        while orography[n]>layer[lev+1]:
        
            var[Nlev-lev-1][n] = NaN
            lev = lev + 1
        #print(n, lev, orography[n],layer[lev-1], orography[n]>layer[lev-1])


def mask_sealand(data,masktype,varname,mask_value=NaN):

    var       = data[varname]
    Nlev      = len (var[:,0])
    Nray      = len (var[0,:])


    if   masktype  == 'show_land'   : mask =      data['landmask']
    elif masktype  == 'show_sea'    : mask = -1*( data['landmask'] - 1)
    elif masktype  == 'show_sealand': mask =  0*( data['landmask']    ) + 1
    elif masktype  == 'show_nothing': mask =  0*( data['landmask']    ) 
    else                            : print('mask type = %s not recognized. Use show_land, show_sea, show_sealand, show_nothing' % masktype ); exit()

    for n in range(Nray):
        lev = 0
        if mask[n] == 0:
            var[:,n] = NaN
        
    

#############################################################################################
#                                                                           #
#############################################################################################



def read_data2D(ncfile,dataset,varname,t=NaN):
    DATA      = {};
    ATTRIBUTE = {};
    print(dataset,varname)
    if varname == 'dummy': 
        return np.empty((1))*NaN, {}
   

    nc_id = netCDF4.Dataset(ncfile,'r');
    nc_attrs, nc_dims, nc_vars = ncdump(nc_id, verb=False)
    #print(varname, varname in nc_vars)
    if varname in nc_vars:
        netcdf4_extract_fields_and_attributes([varname], [ncfile], DATA, ATTRIBUTE);
        data      = DATA     [varname]
        attribute = ATTRIBUTE[varname]
    elif dataset == 'cospout' and varname == 'lwpmodis_plus_iwpmodis':
        varnames = ['lwpmodis','iwpmodis']
        netcdf4_extract_fields_and_attributes(varnames, [ncfile], DATA, ATTRIBUTE);
        data = DATA['lwpmodis'] + DATA['lwpmodis']
        attribute = {}
    else:
        return np.empty((1))*NaN, {}


    # MODIS 
    if dataset == 'modis':
        if varname in nc_vars:
            netcdf4_extract_fields_and_attributes([varname], [ncfile], DATA, ATTRIBUTE);
            data      = DATA     [varname]
            attribute = ATTRIBUTE[varname]
            fillval = attribute['_FillValue']
            data [data == fillval] = NaN
            if varname == 'Cloud_Water_Path': data    = data   * 0.001 # g * m^-2 --> kg * m^-2
        
    # COSPOUT
    elif dataset == 'cospout':
        fillval = -1e+30
        if varname in nc_vars:
            netcdf4_extract_fields_and_attributes([varname], [ncfile], DATA, ATTRIBUTE);
            data      = DATA     [varname]
            attribute = ATTRIBUTE[varname]
            data = data.T
            data [data == fillval] = NaN
            if varname == 'cltmodis'    : data    = data   * 0.01  # %        --> 0-1
            if varname == 'cllmodis'    : data    = data   * 0.01  # %        --> 0-1
            if varname == 'clmmodis'    : data    = data   * 0.01  # %        --> 0-1
            if varname == 'clhmodis'    : data    = data   * 0.01  # %        --> 0-1
            if varname == 'clcalipso'   : data    = data   * 0.01  # %        --> 0-1
            if varname == 'reffclimodis': data    = data   * 1e6
            if varname == 'reffclwmodis': data    = data   * 1e6
            if varname == 'pctmodis'    : data    = data   * 0.01  # To check since data are already supposed to be in hPa
            if varname == 'pctmodis'    : data [data == 0 ] = NaN
        elif 'lwpmodis_plus_iwpmodis':
            varnames = ['lwpmodis','iwpmodis','cltmodis']
            netcdf4_extract_fields_and_attributes(varnames, [ncfile], DATA, ATTRIBUTE);
            data1 = DATA['iwpmodis'].T;
            data2 = DATA['lwpmodis'].T;
            #mask  = DATA['cltmodis'].T;
 
            mask1 = data1 * 0 
            mask2 = data2 * 0 

            mask1 [data1 == fillval] = 1
            mask2 [data2 == fillval] = 1
            mask = mask1 + mask2
            mask [mask <  2] = 1
            mask [mask == 2] = NaN


            data1 [data1 == fillval] = 0
            data2 [data2 == fillval] = 0


            #mask  [mask  == fillval] = NaN;
            #mask = mask * 0 + 1
            


            data = (data1 + data2)*mask
            attribute = ATTRIBUTE['iwpmodis']
    elif dataset == 'mcd06':
        netcdf4_extract_fields_and_attributes([varname], [ncfile], DATA, ATTRIBUTE);
        data      = DATA     [varname]
        attribute = ATTRIBUTE[varname]
        fillval = attribute['_FillValue']
        data [data == fillval] = NaN
        if len(data.shape) == 3: data = data[t,:,:]
        data = data.T
        if 'Cloud_Water_Path' in varname: data    = data   * 0.001 # g * m^-2 --> kg * m^-2
    # GEM
    elif dataset == 'gem':
        if varname in nc_vars:
            netcdf4_extract_fields_and_attributes([varname], [ncfile], DATA, ATTRIBUTE);
            data      = DATA     [varname]
            attribute = ATTRIBUTE[varname]
            if len(data.shape) == 3: data = data[t,:,:]
        
        
    return data, attribute



def resize_data(data, domain):
    if   domain == 'box11'  : grid  = 'NAM-11m.nml'; xi=250; xf=575; yi=205; yf=490;
    output = {}
    for d in data:
        output[d] = data[d][yi:yf,xi:xf];
    return output

#############################################################################################
#                                       Cloudsat modules                                    #
#############################################################################################

############################################################################################
def extract_cloudsat_track(ncfile, coord_domain, timei='19000101000000', timef='22000101000000'):

    if coord_domain == 'global':
        coord_domain = [(-360,-90),  (360,-90), (360,90), (-360,90), (-360,-90)]

    # Extracting latitude and longitude
    varlist    = [ 'longitude', 'latitude']
    data_in    = {}; 
    attributes = {};
    netcdf4_extract_fields_and_attributes(varlist, [ncfile], data_in, attributes);
    
    # Extracting / formating time
    data_in   ['time'] = set_time_cloudsat(ncfile)
    attributes['time'] = {'longname': 'Date: YYYYMMDDhhmmss.f'}

    # Formating cloudsat data (missing, offset, scale factor)
    format_data_cloudsat(data_in, attributes)

    # Constructing cloudsat coord:  (x1,x2...) (y1,y2,...) --> ( (x1,y1), (x2,y2), ... )
    coord_track = construct_coord (data_in['longitude'],data_in['latitude'])

    # Setting a flag (1 = inside, 0 = outside)
    #   for the points inside the polygon    
    #   for the points inside the time range (to implement)
    spatial_flag = point_inside_polygon(coord_track, coord_domain)

    Nray = int(np.sum(spatial_flag))

    data_out = {}
    data_out['longitude'] = np.empty(Nray, dtype=float)
    data_out['latitude' ] = np.empty(Nray, dtype=float)
    data_out['time'     ] = np.empty(Nray, dtype=float)
    data_out['index'    ] = np.empty(Nray, dtype=int)

    j = 0
    for i in range(len(spatial_flag)):
        if (spatial_flag[i] > 0): 
            data_out['longitude'][j] = data_in['longitude'][i]
            data_out['latitude' ][j] = data_in['latitude' ][i]
            data_out['time'     ][j] = data_in['time'     ][i]
            data_out['index'    ][j] = i
            #print(i,spatial_flag[i], data_in['longitude'][i], data_in['latitude'][i], data_in['time'][i])
            j = j + 1



    return data_out

############################################################################################
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

############################################################################################
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



############################################################################################
def format_data_cloudsat(data,attributes):
    for var in data:
        if 'missing' in attributes[var]:
            missing   = attributes[var]['missing']
            missop    = attributes[var]['missop' ]
            #print(missing, missop)
            if missop == '==':
                data[var]  = data[var] * 1.0
                print(var, missing ,data[var].shape)
                data[var] [data[var] == missing ] = NaN
            #else:
            #    print('Missing operator ' + missop + ' not implemented yet')
            #    exit()
        if 'offset' in attributes[var]:
            offset    = attributes[var]['offset']
            factor    = attributes[var]['factor']
            data[var] = (data[var]-offset)/factor

############################################################################################
def construct_profil_cloudsat(ncfile, track, variable):
    data_in    = {};
    attributes = {};
    netcdf4_extract_fields_and_attributes([variable], [ncfile], data_in, attributes);
    format_data_cloudsat(data_in, attributes)
    if len(data_in[variable].shape) == 2:
        data_out = data_in[variable][track['index'],0:105].T
    else:
        data_out = data_in[variable][track['index']]
    return(data_out)
    

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
    return(data_out)









############################################################################################
def point_inside_polygon(points, polygon):
    polygon = Polygon(polygon)
    N       = len(points)
    flag    = np.zeros(N)
    for n in range(N):
        point   = Point(points[n])
        if polygon.contains(point):
            flag[n] = 1
    return flag


###########################################################################################
def construct_coord(x,y):
    coord = []
    N     = len(x)
    for n in range(N):
        coord.append((x[n],y[n]))
    return coord

##########################################################################

def getOverlap(a, b):
    min_a = min(a[0], a[1])
    max_a = max(a[0], a[1])
    min_b = min(b[0], b[1])
    max_b = max(b[0], b[1])

    return max(0, min(max_a, max_b) - max(min_a, min_b))


def compute_overlap_coeff(src_layer, target_layer):
    if len(src_layer.shape) == 1: src_layer = src_layer[:,np.newaxis]
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
                #print(n,lev1,lev2,range1,range2,overlap, total_overlap)
        
            for lev2 in overlap_coeff[N][lev1]:
                overlap_coeff[N][lev1][lev2] = overlap_coeff[N][lev1][lev2] / total_overlap
            
    # PRINT OVERLAP
    if 1 == 0:
        for n in overlap_coeff:
            for lev1 in overlap_coeff[n]:
                print('nray=%4d  target_lev=%4d  [ %8.1f - %8.1f  ]:    ' % (n, lev1,target_layer[lev1],target_layer[lev1+1]), end = ''  ) 
                for lev2 in overlap_coeff[n][lev1]:
                    print('%4d [ %8.1f - %8.1f  ]: %3.2f  ' % (lev2,src_layer[lev2,n],src_layer[lev2+1,n], overlap_coeff[n][lev1][lev2]), end = ''  )
                print('')
    return overlap_coeff


def format_levels(field, overlap_coeff,Nlev):
    #Nlev   = 105
    Nray  = field.shape[1]
    Ncoef = len(overlap_coeff)
    output = np.zeros((Nlev,Nray))
    for n in range(Nray):
        
        for lev1 in range(Nlev):
            if    Ncoef == 1: olc = overlap_coeff[0][lev1] 
            else            : olc = overlap_coeff[n][lev1]
            #print(n,Nray,lev1,Nlev,overlap_coeff[n][lev1])

            if bool(olc):
                value = 0
                for lev2 in olc:
                    coeff = olc[lev2]
                    value = value + coeff * field[lev2,n]
            
            #if bool(overlap_coeff[n][lev1]):
                #value = 0
                #for lev2 in overlap_coeff[n][lev1]:
                #    coeff = overlap_coeff[n][lev1][lev2]
                #    value = value + coeff * field[lev2,n]
            else:
                value = NaN
            output[lev1][n] = value
    return output

