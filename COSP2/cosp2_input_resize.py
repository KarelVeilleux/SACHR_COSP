# info: poitras.vincent@uqam.ca
# date: 2022/02/18
# aim : Reduce the size of the cosp_input field


import numpy as np
import netCDF4

import sys;                      sys.path.append('/home/poitras/SCRIPTS/mes_modules_python')
from   netcdf4_extra             import netcdf4_extract_fields_and_attributes


###############################################################################################
lati  = int(sys.argv[1])
latf  = int(sys.argv[2])
loni  = int(sys.argv[3])
lonf  = int(sys.argv[4])
levi  = int(sys.argv[5])
levf  = int(sys.argv[6])
filei = sys.argv[7]
fileo = sys.argv[8]





#filei = '/pampa/poitras/DATA/TREATED/COSP_INPUT/CORDEX/Cascades_CORDEX/CLASS/NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/storm_019_q1_q2_q3q4q5q6/cosp_input_201501051900.nc'
#fileo = '/home/poitras/SCRIPTS/COSPv2.0/driver3/data/P3/inputs/cosp_input_201401051900_q1_q2_q3q4q5q6_rz.nc'
#lati=205
#latf=490
#loni=250
#lonf=575
#levi=0
#levf=71



###############################################################################################
#Reading data

ncdata = netCDF4.Dataset(filei,'r');
ncfile = [ filei ]
data       = {};
attributes = {};
netcdf4_extract_fields_and_attributes('all_variables', ncfile, data, attributes);


##############################################################################################
#Output data

ncfile = netCDF4.Dataset(fileo, 'w', format='NETCDF4')

# Create diemsnion
ncfile.createDimension('lon'  ,lonf-loni)
ncfile.createDimension('lat'  ,latf-lati)
ncfile.createDimension('level',levf-levi)
ncfile.createDimension('hydro',   9)

# Create variable: lat, lon
ncfile.createVariable('lon', 'f4', ('lon'))
ncfile['lon'][:] = ncdata['lon'][loni:lonf]
ncfile.createVariable('lat', 'f4', ('lat'))
ncfile['lat'][:] = ncdata['lat'][lati:latf]

for fieldname in  data:
    #print(fieldname)
    data[fieldname] = np.nan_to_num(data[fieldname])
    
    fieldshape = data[fieldname].shape
    if len(fieldshape) == 0:        # emsfc_lw
        ncfile.createVariable(fieldname, 'f4',zlib=True)
        ncfile[fieldname][:] = data[fieldname]

    elif len(fieldshape) == 2:      # sunlit, skt, orography, landmask
        ncfile.createVariable(fieldname, 'f4', ('lat','lon'),zlib=True)
        ncfile[fieldname][:] = data[fieldname][lati:latf,loni:lonf]

    elif len(fieldshape) == 3:      # all other variables
        ncfile.createVariable(fieldname, 'f4', ('level','lat','lon'),zlib=True)
        ncfile[fieldname][:] = data[fieldname][levi:levf,lati:latf,loni:lonf]

    elif len(fieldshape) == 4:      # effective radii
        ncfile.createVariable(fieldname, 'f4', ('hydro','level','lat','lon'),zlib=True)
        ncfile[fieldname][:] = data[fieldname][:,levi:levf,lati:latf,loni:lonf]
    
    ncfile[fieldname].setncatts(ncdata[fieldname].__dict__)

#print(fileo)
