import os
import sys
import yaml
import netCDF4
import warnings
import yaml
import numpy  as np
import pandas as pd
import math

NaN = np.nan

#################################################################################################################
### FUNCTIONS ###################################################################################################
#################################################################################################################
def create_dataframe(dirlist, YYYYMM):
    # Create a dataframe containg only the files of YYYYMM 
    YYYY =     str(YYYYMM[0:4])
    MM   = int(str(YYYYMM[4:6]))

    filelist   = dirlist + '/' + YYYY + '.txt'
    df         = pd.read_csv(filelist,delimiter='\s+', header=None)
    df.columns = ['file', 'ndata', 'ti', 'tf', 'date', 'MM', 'date_gem', 't_gem']
    df         = df[   np.isin(df['MM'], MM)  ].reset_index(drop=True)
    return df

# Random Overlap  
def random_overlap(nc_array):
    ran_overlap = np.zeros(nc_array.shape[0])

    for index,row in pd.DataFrame(nc_array).iterrows():
        one_minus_cover = np.array([1 - (x) for x in row])
        filtered        = one_minus_cover[~np.isnan(one_minus_cover)]
        ran_overlap[index] = 1 - math.prod(filtered)
        
    return ran_overlap

def compute_cloud_cover(ncfilei, layerdef):
###############################################################################################################################
# To compute Total, High, Mid and Low cloud cover
#        High, Mid and Low cloud cover may be defined :
#            - From altitude (layerdef = amax, amin, amean)
#            - From pressure (layerdef = pmax, pmin, pmean)
#        All levels of the profiles beloging to a layer (total, high, mid and low) may be 
#        combined using (we used a "toy" example with 3 levels of cloud fraction to illustrate the definitions.
#            - With : C1 = 0.3 ; C2 = 0.2 ; C3 = 0.4
#             
#                Levels
#                     ^
#                   1 |-----                 
#                   2 |    ----
#                   3 |        ------
#                     |- - - - -|- - - - ->
#                     0        0.5        1
#                         Cloud fraction
#
#            - Max overlap     = max{C1,C2,C3}              = 0.4                   
#            - Min overlap     = min{1,C1 + C2 + C3}        = 0.9         
#            - Random overlap  = 1-(1 - C1)(1 - C2)(1 - C3) = 0.66       
#            - Mean overlap    = mean{C1,C2,C3}             = 0.3
#            
#            For further information, see General equation (4) in :
#            Oreopoulos, L., & Khairoutdinov, M. (2003). Overlap properties of clouds generated by a cloud‐resolving model. 
#            Journal of Geophysical Research: Atmospheres, 108(D15).  
#                 
#            - Max-Ran overlap = 1 - (1-C1)X{((1-max(C1,C2))/(1-C1))•((1-max(C2,C3))/(1-C2))} = 0.47
###############################################################################################################################

    # Reading the NetCDF file
    nc = netCDF4.Dataset(ncfilei,'r')

    # Création de masques pour garder les points du profile pour une couche donnée (Low,Mid,High)
    if layerdef[0] == 'p':
        Pressure  = nc['Pressure'][:]
        mask_high = np.where(Pressure <  pressure_lim_midhigh,  1, NaN)
        mask_low  = np.where(Pressure >= pressure_lim_midlow ,  1, NaN)
        mask_mid  = np.where(Pressure >= pressure_lim_midhigh,  1, NaN) * np.where(Pressure < pressure_lim_midlow , 1, NaN)
    elif layerdef[0] == 'a':
        Altitude = np.flip(np.arange(-480,-480+(398+1)*60,60))  # Il y a 398 niveaux de 60m commençant à -480 m
        mask_high = np.where(Altitude >  altitude_lim_midhigh, 1, NaN)
        mask_low  = np.where(Altitude <= altitude_lim_midlow , 1, NaN)
        mask_mid  = np.where(Altitude <= altitude_lim_midhigh, 1, NaN) * np.where(Altitude > altitude_lim_midlow , 1, NaN)


    Cloud_Layer_Fraction      = nc['Cloud_Layer_Fraction'][:] / 30  # On divise par 30 car il ya 30 mesures par points de grille
    Cloud_Layer_Fraction_high = Cloud_Layer_Fraction * mask_high
    Cloud_Layer_Fraction_mid  = Cloud_Layer_Fraction * mask_mid
    Cloud_Layer_Fraction_low  = Cloud_Layer_Fraction * mask_low

    #Merging" the cloud_cover from each level
    # C_max = max(C_1, C_k)'
    if 'max' in layerdef:
        Cloud_Cover      = np.nanmax(Cloud_Layer_Fraction     ,axis=1)                                                             
        Cloud_Cover_high = np.nanmax(Cloud_Layer_Fraction_high,axis=1)
        Cloud_Cover_mid  = np.nanmax(Cloud_Layer_Fraction_mid ,axis=1)
        Cloud_Cover_low  = np.nanmax(Cloud_Layer_Fraction_low ,axis=1)
    # C_min = max(1,C1 +...+ C_k)
    elif 'min' in layerdef:
        Cloud_Cover      = np.nansum(Cloud_Layer_Fraction     ,axis=1); Cloud_Cover      = np.where(Cloud_Cover      <= 1, Cloud_Cover     , 1)
        Cloud_Cover_high = np.nansum(Cloud_Layer_Fraction_high,axis=1); Cloud_Cover_high = np.where(Cloud_Cover_high <= 1, Cloud_Cover_high, 1)
        Cloud_Cover_mid  = np.nansum(Cloud_Layer_Fraction_mid ,axis=1); Cloud_Cover_mid  = np.where(Cloud_Cover_mid  <= 1, Cloud_Cover_mid , 1)
        Cloud_Cover_low  = np.nansum(Cloud_Layer_Fraction_low ,axis=1); Cloud_Cover_low  = np.where(Cloud_Cover_low  <= 1, Cloud_Cover_low , 1)
    # C_ran = 1-Product{1-C_i}, with i from 1 to k
    elif 'ran' in layerdef:
        Cloud_Cover      = random_overlap(Cloud_Layer_Fraction     )                                                             
        Cloud_Cover_high = random_overlap(Cloud_Layer_Fraction_high)
        Cloud_Cover_mid  = random_overlap(Cloud_Layer_Fraction_mid )
        Cloud_Cover_low  = random_overlap(Cloud_Layer_Fraction_low )       
    # C_maxran = 1 - (1 - C_1)X Product{(1-max(C_i-1, C_i))/(1 - C_i-1)}
    #elif 'maxran' in layerdef:
        
    # C_mean = mean(C_1, C_k) 
    elif 'mean' in layerdef:
        Cloud_Cover      = np.nanmean(Cloud_Layer_Fraction     ,axis=1)
        Cloud_Cover_high = np.nanmean(Cloud_Layer_Fraction_high,axis=1)
        Cloud_Cover_mid  = np.nanmean(Cloud_Layer_Fraction_mid ,axis=1)
        Cloud_Cover_low  = np.nanmean(Cloud_Layer_Fraction_low ,axis=1)

    return Cloud_Cover, Cloud_Cover_high, Cloud_Cover_mid, Cloud_Cover_low

def create_ouput_file(ncfilei,ncfileo,Tot_cloud_cover, High_cloud_cover, Mid_cloud_cover, Low_cloud_cover, layerdef):
    
    # I/O file 
    if     os.path.exists(ncfileo)                 : os.remove(ncfileo)
    if not os.path.exists(os.path.dirname(ncfileo)): os.makedirs(os.path.dirname(ncfileo))
    nci = netCDF4.Dataset(ncfilei, 'r');
    nco = netCDF4.Dataset(ncfileo, 'w');


    # New data to output / Old data to copy
    newvarnames = [ 'Tot_cloud_cover', 'High_cloud_cover', 'Mid_cloud_cover', 'Low_cloud_cover']
    varnames    = [ 'Profile_Time', 'Profile_UTC_Time', 'Latitude', 'Longitude', 'Pressure', 'Day_Night_Flag' ]
    
    # Extracting the diemsnion
    dimnames = []
    for varname in varnames:
        for dimname in nci.variables[varname].dimensions:
            dimnames.append(dimname)
    dimnames = set(dimnames)

    # Copying global attributes
    nco.setncatts(nci.__dict__)

    # Copying dimensions
    # dimnames = [dim for dim in nci.dimensions]
    for dimname in dimnames:
        dimension = nci.dimensions[dimname]
        nco.createDimension(dimname, (len(dimension) if not dimension.isunlimited() else None))

    # Copying (old) variables
    for varname in varnames:
        variable = nci[varname]
        x = nco.createVariable(varname, variable.datatype, variable.dimensions, zlib=True, complevel=4)
        nco[varname].setncatts(variable.__dict__)
        nco[varname][:] = nci[varname][:]

    # Long_name for the new variable
    if    'max'  in layerdef: str_layerdef     = 'maximum overlap'
    elif  'min'  in layerdef: str_layerdef     = 'minimum overlap'
    elif  'ran'  in layerdef: str_layerdef     = 'random overlap'
    #elif  'maxran'  in layerdef: str_layerdef = 'max-ran overlap'

    elif  'mean' in layerdef: str_layerdef = 'mean value'
    if layerdef[0] == 'p':
        long_name_Tot  = 'Total cloud cover: %s (all layers)'                     % (str_layerdef)
        long_name_High = 'High cloud cover : %s (layers with p < %dha)'           % (str_layerdef, pressure_lim_midhigh)
        long_name_Mid  = 'Mid cloud cover  : %s (layers with %dhPa > p >= %dhPa)' % (str_layerdef, pressure_lim_midlow, pressure_lim_midhigh)
        long_name_Low  = 'Low cloud cover  : %s (layers with p >= %dhPa)'         % (str_layerdef, pressure_lim_midlow)
    elif layerdef[0] == 'a':
        long_name_Tot  = 'Total cloud cover: %s (all layers)'                     % (str_layerdef)
        long_name_High = 'High cloud cover : %s (layers with h > %dm)'            % (str_layerdef, altitude_lim_midhigh)
        long_name_Mid  = 'Mid cloud cover  : %s (layers with %dm < h <= %dm )'    % (str_layerdef, altitude_lim_midlow, altitude_lim_midhigh)
        long_name_Low  = 'Low cloud cover  : %s (layers with h <= %dm)'           % (str_layerdef, altitude_lim_midlow)


    # Writting new variables
    for varname in newvarnames:

        if   varname ==  'Tot_cloud_cover' : data =  Tot_cloud_cover; long_name = long_name_Tot
        elif varname == 'High_cloud_cover' : data = High_cloud_cover; long_name = long_name_High
        elif varname ==  'Mid_cloud_cover' : data =  Mid_cloud_cover; long_name = long_name_Mid
        elif varname ==  'Low_cloud_cover' : data =  Low_cloud_cover; long_name = long_name_Low
        
        x = nco.createVariable(varname, 'float32', 'fakeDim0', zlib=True, complevel=4)
        nco[varname].setncatts({'long_name': long_name})
        nco[varname][:] = data[:]

#################################################################################################################
### MAIN ########################################################################################################
#################################################################################################################


#########################################################################
# Input arguments                                                       #
#########################################################################
working_directory =     sys.argv[1]
YYYYMM            =     sys.argv[2]
layerdef          =     sys.argv[3]
overwrite         =     sys.argv[4].lower()

YYYY = int(str(YYYYMM[0:4]))
MM   = int(str(YYYYMM[4:6]))

#########################################################################
# Configuration file (yml)                                              #
#########################################################################
yml_file = working_directory + '/../config.yml'
stream = open(yml_file,'r')
config = yaml.safe_load(stream)

domain  = config['domain' ]
dirout  = config['CALIPSO_CHINOOK']['NetCDF'] + '_LowMidHigh/' + layerdef
dirlist = config['CALIPSO']['list']   + '/' + domain

if not os.path.exists(dirout): os.makedirs(dirout)

#########################################################################
# Hardcoded parameter                                                   #
#########################################################################

# Altitude and pressure limit for High, Mid, Low layer.                       #
# See eg: https://climserv.ipsl.polytechnique.fr/cfmip-obs/Calipso_goccp.html #
pressure_lim_midlow  =  680 # hPa
pressure_lim_midhigh =  440 # hPa
    
altitude_lim_midlow  = 3200 # m 
altitude_lim_midhigh = 6500 # m 

#########################################################################
# Computing Low, Mid, High, Tot cloud cover                             #
#########################################################################

#Create a dataframe containing YYYYMM files
df_CALIPSO = create_dataframe(dirlist, YYYYMM)

#Loop over each file of the dataframe
for iCAL in range(len(df_CALIPSO['file'])):

    # Compute Tot_cloud_cover, High_cloud_cover, Mid_cloud_cover, Low_cloud_cover using layer definition given by layerdef
    date    = df_CALIPSO['date'][iCAL]
    ncfilei = df_CALIPSO['file'][iCAL]
    Tot_cloud_cover, High_cloud_cover, Mid_cloud_cover, Low_cloud_cover = compute_cloud_cover(ncfilei, layerdef)
    
    # Output data
    YYYY    = os.path.basename(os.path.dirname(ncfilei))
    ncfileo = dirout + '/' + YYYY + '/' + os.path.basename(ncfilei)     
    create_ouput_file(ncfilei, ncfileo, Tot_cloud_cover, High_cloud_cover, Mid_cloud_cover, Low_cloud_cover, layerdef)
    print(date, ncfileo)


