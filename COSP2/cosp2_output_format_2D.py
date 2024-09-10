# info: poitras.vincent@uqam.ca
# date: 2022-02-18
# aim : convert cosp output files (NetCDF): loc --> (lat,lon)

import netCDF4
import numpy   as np
import sys

################################################################################################################################
#                                                           File paths (to edit)                                               #
################################################################################################################################
# Note: nc_grid is facultative, it is used only to provided information relatated to the projection
# Set nc_dgrid = 'nofile' if you don't want to use it


nc_input  = sys.argv[1]
nc_output = sys.argv[2]
nc_grid   = sys.argv[3]


#dirout    = '/pampa/poitras/DATA/TREATED/COSP2/Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/MPB/OUTPUT/1M001SC'
#nc_input  = dirout + '/' 'cosp_output_201401051900_cs_test2.nc'
#nc_output = dirout + '/' 'cosp_output_201401051900_cs_test2_2D.nc'
#nc_grid   = '/pampa/poitras/SCRATCH/COSP2/cospin_test.nc'
#print(nc_output)
################################################################################################################################
#                                                             MODULES                                                          #
################################################################################################################################
def reshape_field(field,field_dimensions,dimensions_size):
    
    number_of_field_dimensions = len(field_dimensions)
    lon_size = dimensions_size['lon'] 
    lat_size = dimensions_size['lat']

    dim = {}
    shape_array = []
    for i in range(number_of_field_dimensions-1):
        dim[i] = dimensions_size[field_dimensions[i]]
        shape_array = shape_array + [ dim[i] ]

    shape_array = shape_array + [ lon_size ]
    shape_array = shape_array + [ lat_size ]
    reshaped_field = np.empty(shape_array)


    if number_of_field_dimensions   == 1:
        reshaped_field = np.reshape(field, (lon_size, lat_size), order='F')

    elif number_of_field_dimensions == 2:
        for i1 in range(dim[0]):
            reshaped_field[i1,:,:] = np.reshape(field[i1,:], (lon_size, lat_size), order='F')

    elif number_of_field_dimensions == 3:
        for i1 in range(dim[0]):
            for i2 in range(dim[1]):
                reshaped_field[i1,i2,:,:] = np.reshape(field[i1,i2,:], (lon_size, lat_size), order='F')

    elif number_of_field_dimensions == 4:
        for i1 in range(dim[0]):
            for i2 in range(dim[1]):
                for i3 in range(dim[2]):
                    reshaped_field[i1,i2,i3,:,:] = np.reshape(field[i1,i2,i3,:], (lon_size, lat_size), order='F')

    elif number_of_field_dimensions == 5:
        for i1 in range(dim[0]):
            for i2 in range(dim[1]):
                for i3 in range(dim[2]):
                    for i4 in range(dim[3]):
                        reshaped_field[i1,i2,i3,i4,:,:] = np.reshape(field[i1,i2,i3,i4,:], (lon_size, lat_size), order='F')
    else:
        reshaped_field = field[:]

    return reshaped_field



################################################################################################################################
#                                                             MAIN PART                                                        #
################################################################################################################################
# Opening nc-files
src = netCDF4.Dataset(nc_input , 'r');
dst = netCDF4.Dataset(nc_output, 'w', format='NETCDF4') 



# Extracting size of each dimension
dimensions_size              = {}
dimensions_size['lon'] = len( set(src['longitude'][:])) 
dimensions_size['lat'] = len( set(src['latitude' ][:]))

for dimension  in src.dimensions:
    dimensions_size[dimension] = len(src.dimensions[dimension])



# Reshaping the fields
reshaped_fields            = {}
reshaped_fields_dimensions = {}

for name in src.variables:
    field = src[name]
    field_dimensions = field.dimensions
    if 'loc' in field_dimensions:       
        reshaped_fields           [name] = reshape_field(field, field_dimensions, dimensions_size)
        reshaped_fields_dimensions[name] = field_dimensions[:-1] + ('lon','lat') 
    else:
        reshaped_fields           [name] = field[:]
        reshaped_fields_dimensions[name] = field_dimensions

reshaped_fields['lon'] = np.sort(np.unique(src.variables['longitude'][:]))
reshaped_fields['lat'] = np.sort(np.unique(src.variables['latitude' ][:]))
reshaped_fields_dimensions['lon'] = ('lon')
reshaped_fields_dimensions['lat'] = ('lat')


# Copying global attributes all at once via dictionary (turned of it was causing problem with the geolocation)
#dst.setncatts(src.__dict__)


# Copying existing dimensions (except loc)
for name, dimension in src.dimensions.items():
    if name != 'loc':
        dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))


# Creating new dimension longitude and latitude (loc --> (lon, lat)
dst.createDimension('lat', dimensions_size['lat'])
dst.createDimension('lon', dimensions_size['lon'])


# Creating output fields
for name in src.variables:
    variable = src[name]
    x = dst.createVariable(name, variable.datatype, reshaped_fields_dimensions[name],zlib=True)
    dst[name][:] = reshaped_fields[name]
    dst[name].setncatts(variable.__dict__)

x = dst.createVariable('lon', 'f4', reshaped_fields_dimensions['lon'],zlib=True)
x = dst.createVariable('lat', 'f4', reshaped_fields_dimensions['lat'],zlib=True)
dst['lon'][:] = reshaped_fields['lon']
dst['lat'][:] = reshaped_fields['lat']



# Adding projection information (facultative)
if nc_grid != 'nofile':
    grd   = netCDF4.Dataset(nc_grid , 'r');
    x = dst.createVariable('rotated_pole', grd['rotated_pole'].datatype )
    dst['rotated_pole'].setncatts(grd['rotated_pole'].__dict__)
    dst['lon'         ].setncatts(grd['lon'         ].__dict__)
    dst['lat'         ].setncatts(grd['lat'         ].__dict__)
    for name in src.variables:
        dst[name].setncatts({'grid_mapping': 'rotated_pole'})    
    

print(nc_output)




