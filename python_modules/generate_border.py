import numpy    as     np
import os
import sys;            sys.path.append('/home/poitras/SCRIPTS/mes_modules_python')
from   read_shapefile  import rshapefile
from   grid_projection import read_gem_settings              as readgem
from   grid_projection import convert_rotlatlon_to_cartesian as gridcon
NaN = np.nan


def generate_border (nml, gridtype, shapefile, shift=[0,0], extension=[0,0]):
    if isinstance(nml, str):
        if os.path.isfile(nml)  :   grd = readgem(nml)
        else                    :   grd = readgem('/chinook/poitras/gem_settings/'+nml)
    else:
        grd = nml

    #Free grid
    xlim = [-0.5, grd['ni']-0.5-2*grd['blend_H']];
    ylim = [-0.5, grd['nj']-0.5-2*grd['blend_H']];
    lim = { 'xlim': xlim, 'ylim': ylim}
 
    lon, lat = rshapefile(shapefile)
    i, j    = gridcon( lon, lat, grd, gridtype, 'lonlat2index')
    i[i >  xlim[1]+extension[0]] = NaN; j[j >  ylim[1]+extension[1]] = NaN;
    i[i <  xlim[0]-extension[0]] = NaN; j[j <  ylim[0]-extension[1]] = NaN;
    border = { 'i': i+shift[0], 'j': j+shift[1] }


    return border, lim

