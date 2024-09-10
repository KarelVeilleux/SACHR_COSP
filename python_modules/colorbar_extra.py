import numpy as np
NaN = np.nan

def set_colorbar_extend(data,clim):
    if   np.nanmin(np.nanmin(data))>clim[0] and np.nanmax(np.nanmax(data))>clim[1]: extend='max'
    elif np.nanmin(np.nanmin(data))<clim[0] and np.nanmax(np.nanmax(data))<clim[1]: extend='min'
    elif np.nanmin(np.nanmin(data))<clim[0] and np.nanmax(np.nanmax(data))>clim[1]: extend='both'
    else:                                                                           extend='neither'

    return extend
