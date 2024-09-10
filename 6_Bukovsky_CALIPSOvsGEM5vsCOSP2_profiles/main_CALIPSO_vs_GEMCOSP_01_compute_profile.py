import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd
import calendar
import yaml
import sys
import os
import netCDF4
import glob
import xarray as xr


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aux_CALIPSO         import extract_domain_coord
from aux_CALIPSO         import extract_satellite_track
from aux_grid_projection import convert_latlon_to_domain_indices
from auxilliary_functions import get_args


sys.path.append('/home/veilleux/Projet/Projet_SACHR/analyses_and_figures/python_modules')
from domain    import generate_domain_coord
from domain    import convert_latlon_to_domain_indices


import warnings
warnings.filterwarnings("ignore")




def format_levels(datain, overlap_coeff, Nlevout):
    Nray      = datain.shape[0]
    Nlevin    = datain.shape[1]
    Nray_coef = len(overlap_coeff)     # Nray_coeff = 1 pour CALIPSO et COSP, car le profil vertical (et donc les coefficients)  ne change pas
    dataout   = np.zeros([Nray,Nlevout])
    for n in range(Nray):
        for levout in range(Nlevout-1):  
            d = 0
            if    Nray_coef == 1: olc = overlap_coeff[0][levout]
            else                : olc = overlap_coeff[n][levout]
            for levin in olc:
                if format_level_method == 'maximum_overlap':
                    if  datain[n,Nlevin-levin-1]>d:                             
                        dataout[n,Nlevout-1-levout] = datain[n,Nlevin-levin-1]
                elif format_level_method == 'minimum_overlap':
                    print('minimum_overlap method is not implemented')
                    exit()
                elif format_level_method == 'random_maximum_overlap':
                    print('random_maximum_overlap method is not implemented')
                    exit()
                elif format_level_method == 'weighted_mean':
                    dataout[n,Nlevout-1-levout] +=  olc[levin] * datain[n,Nlevin-levin-1] 
                #if datain[n,levin] > 0 and n == 0:
                #    print('n=%3d levout=%3d (levin=%3d): %7.3f * %7.3f = %7.3f, tot=%7.3f' % (n, levout, levin, olc[levin], datain[n,levin] , olc[levin] * datain[n,levin] , dataout[n,levout]))    
    return dataout



def getOverlap(a, b):    
    min_a = min(a[0], a[1])
    max_a = max(a[0], a[1])
    min_b = min(b[0], b[1])
    max_b = max(b[0], b[1])
    return max(0, min(max_a, max_b) - max(min_a, min_b))


def compute_overlap_coeff(src_layer, target_layer):    
    if len(src_layer.shape) == 1: 
        src_layer = src_layer[:,np.newaxis]

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
                # print(n, end='');
                # print('  %4d: [%6.2f,%6.2f]' % (lev1, range1[0], range1[1]), end='')
                # print('  %4d: [%6.2f,%6.2f]' % (lev2, range2[0], range2[1]), end='')
                # print('  %f %f' % (overlap, total_overlap))                        
            for lev2 in overlap_coeff[n][lev1]:
                overlap_coeff[n][lev1][lev2] = overlap_coeff[n][lev1][lev2] / total_overlap
    return overlap_coeff



def convert_calipso_data_in_2D(ncfile, varname, domain, coord_domain, dim):
    ''' Extract the part of the track that is INSIDE the domain '''
    ''' More than 1 "ray" may be located inside a "pixel", this is why we are computing a mean '''
    track = extract_satellite_track(ncfile, coord_domain, showtrack=False)
    indices = convert_latlon_to_domain_indices(track, domain)
    ds      = netCDF4.Dataset(ncfile,'r')
    data    = ds[varname][track['index']]
    dim       = [data.shape[1], dim[1], dim[2]]  # set number of vertical level. dim(data) = nray x nlvel
    data_sum  = np.ones(dim) * 0
    data_n    = np.ones(dim) * 0
    Nray = len(indices['i'])
    for n in range(Nray):
        I = indices['i'][n]
        J = indices['j'][n]
        data_sum[:, I, J] = data_sum[:, I, J] + data[n]
        data_n  [:, I, J] = data_n  [:, I, J] + 1
    return data_sum/data_n



def create_output(column_formatted_sum, column_formatted_n, layerout, dirout, bukovsky_region, YYYYMM):
    fileout = "%s/%s_%s/%s/txt/%s.txt" % (dirout, layerout_dataset, format_level_method, bukovsky_region, YYYYMM)
    if not os.path.exists(os.path.dirname(fileout)): os.makedirs(os.path.dirname(fileout))
    f = open(fileout, 'w')

    # Header
    print('altitude\t', end='', file=f)
    for dataset in column_formatted_sum:
        print('%s_sum\t%s_n\t' % (dataset,dataset), end='', file=f)
    print('', file=f)
    
    
    # Data
    for n,layer in enumerate(np.flipud(layerout)):
        print('%f\t' % layer, end='',file=f)
        for dataset in column_formatted_sum:
            print('%f\t%f\t' % (column_formatted_sum[dataset][n], column_formatted_n[dataset][n]), end='', file=f)

        print('', file=f)
        n += 1
    f.close()
    print(fileout)
 
def main():

    # Parsed args
    args = ['configuration','year','month']
    for arg, value in get_args(sys.argv).items():
        if arg in args:
            globals()[arg] = value
            print(arg, value)

    # Configuration
    stream       = open(configuration,'r')
    config       = yaml.safe_load(stream)
    domain       = config['domain'  ]
    bukovsky     = config['bukovsky']['chinook']
    CALIPSO_list = config['CALIPSO' ]['list'  ]
    COSP2_input  = config['COSP2'   ]['input' ]
    COSP2_output = config['COSP2'   ]['output']
    GEM5_NetCDF  = config['GEM5'    ]['NetCDF']
    dirout       = config['CALIPSOvsCOSP2vsGEM5_maps']['Profiles']

    # Selecting files for the desired year/month from the file list (--> CALIPSO is passing over the domain)
    last_day_of_the_month = calendar.monthrange(year, month)[1]
    YYYYMMDDi = int('%4d%02d%02d' % (year,month,1))
    YYYYMMDDf = int('%4d%02d%02d' % (year,month,last_day_of_the_month))
    list_txt  = CALIPSO_list + '/' + domain + '/' + str(year) + '.txt'
    list_df   = pd.read_csv(list_txt,delimiter='\s+', header=None)


    list_df.columns      = ['nc_CALIPSO', 'ndata', 'ti', 'tf', 'date', 'MM', 'date_gem', 't_gem']
    list_df['YYYYMMDD']  = [int(x[0:8]) for x in list_df['date'].astype(str)]
    list_df              = list_df[   list_df['YYYYMMDD'] >= YYYYMMDDi].reset_index(drop=True)
    list_df              = list_df[   list_df['YYYYMMDD'] <= YYYYMMDDf].reset_index(drop=True)
    #list_df              = list_df.loc[list_df['nc_CALIPSO'].str.contains('D.nc')].reset_index(drop=True)

    print(list_df) 
     

    # Loop on Bukovski regions (ça serait sans doute de déplacer cette boucle à l'intérieur de celle usr la liste, mais ça demanderait un certain travail...)
    for bukovsky_region in bukovsky_regions:

        # bukovsky mask
        file_bukovsky = bukovsky + '/final/' + bukovsky_region + '.nc'
        mask_bukovsky = np.squeeze(xr.open_dataset(file_bukovsky)['mask'].values)
   
        column_sum = {}
        column_n   = {}
        column_formatted_sum = {}
        column_formatted_n   = {}

        data   = {}
        layers = {}

        # loop on each timestep
        for index, row in list_df.iterrows():
            
            YYYYMMDDhhmm = str(row['date'])
            YYYYMM       = str(row['date'])[0:6]
            YYYYMMDD_gem = str(row['date_gem'])
            YYYYMM_gem   = str(row['date_gem'])[0:6]
            print(bukovsky_region, YYYYMMDDhhmm) 


            ######################################
            # Get data + layers for each dataset #
            ######################################
            ncfile_COSPOUT = '%s/%s/M01SC002/CALIPSO/calipso_cloudprofile/%s/cospout_%s_2D.nc' % (COSP2_output, domain, YYYYMM, YYYYMMDDhhmm) 
            ncfile_CALIPSO = row['nc_CALIPSO']
            ncfile_GEM5    = glob.glob('%s/*%s/pm*%sd.nc' % (GEM5_NetCDF, YYYYMM_gem, YYYYMMDD_gem))[0]
            ncfile_COSPIN  = '%s/%s/%s.nc' % (COSP2_input, YYYYMM, YYYYMMDDhhmm)
        
            if  os.path.exists(ncfile_COSPOUT) and os.path.exists(ncfile_CALIPSO) and os.path.exists(ncfile_GEM5) and os.path.exists(ncfile_COSPIN):

                ### COSP2 output ###
                file_COSPOUT      = '%s/%s/M01SC002/CALIPSO/calipso_cloudprofile/%s/cospout_%s_2D.nc' % (COSP2_output, domain, YYYYMM, YYYYMMDDhhmm)
                ds_COSPOUT        = xr.open_dataset(file_COSPOUT)
                data_COSPOUT      = ds_COSPOUT['clcalipso'].values.transpose(0, 2, 1) / 100         # range [0,100] --> [0,1]
                data  ['COSPOUT'] = np.where(data_COSPOUT<0, 0, data_COSPOUT) 
                layers['COSPOUT'] = np.flipud(ds_COSPOUT['levStat'].values/1000) - ds_COSPOUT['levStat'].values[-1]/1000    
                print(file_COSPOUT)

                ### COSP2 input ###
                file_COSPIN             = '%s/%s/%s.nc' % (COSP2_input, YYYYMM, YYYYMMDDhhmm)
                ds_COSPIN               = xr.open_dataset(file_COSPIN)
                data  ['COSPIN']        = np.flipud(ds_COSPIN['tca'   ].values )
                layers_COSPIN_2DxNlevin = ds_COSPIN['height'].values / 1000
                print(file_COSPIN)

                ### GEM5 ### 
                ''' On utilise COSPIN au lieu de GEM5 car les hauteurs ont déjà été calculé  dans COSPIN 
                    Les fichiers de GEM 5 on des pas de temps, alors que pour COSPIN il y a un seul pas de temps
                    la fonction extract_coord_dom utilisé plus bas, s'attends à avoir des pas de temps, c,esy pourquoi on utilise file_GEM5
                    Il faudrait modifier la fonction extract_coord_dom pour pouvoir utilisé file_COSPIN ...
                    Autrement la couverture nuageuse est la même
                '''
                file_GEM5    = glob.glob('%s/*%s/pm*%sd.nc' % (GEM5_NetCDF, YYYYMM_gem, YYYYMMDD_gem))[0]
                ds_GEM5      = xr.open_dataset(file_GEM5).isel(time=row['t_gem'])
                data['GEM5'] = ds_GEM5 ['FN'].values

                ### CALISPO ###
                file_CALISPO = row['nc_CALIPSO']
                if index == 0:  
                    coord_domain = extract_domain_coord(file_GEM5)
                    dim          = data['COSPIN'].shape
                data_CALIPSO = convert_calipso_data_in_2D(row['nc_CALIPSO'], 'Cloud_Layer_Fraction', domain, coord_domain, dim) 
                data  ['CALIPSO'] = data_CALIPSO / 30 # range: [0,30] --> [0,1]
                layers['CALIPSO'] = np.arange(-0.5,-0.5+399*0.06,0.06)
                # https://asdc.larc.nasa.gov/documents/calipso/quality_summaries/CALIOP_L2ProfileProducts_2.01.pdf
                #          The cloud profile products are reported at a uniform spatial resolution of 60-m vertically and 
                #          5-km horizontally, over a nominal altitude range from 20-km to -0.5-km
                print(file_CALISPO)
                print('')

                layerout = layers[layerout_dataset]
                Nlevout  = len(layerout)
                overlap_coeff = {}
                for dataset in datasets:
                    if dataset in layers:
                        overlap_coeff[dataset] = compute_overlap_coeff(layers[dataset], layerout)



                #####################
                # Mask              #
                #####################
                mask_track = np.where(np.isnan(data_CALIPSO).any(axis=0) == 0, 1 , np.nan)
                mask       = mask_track * mask_bukovsky


                # Validation de l'orientation de données/mask (pour le développement)
                if False:
                    plt.figure(1); plt.imshow(np.sum(data['COSPOUT'  ], axis=0),interpolation='none'); plt.gca().invert_yaxis(); plt.title('COSPOUT')
                    plt.figure(12); plt.imshow(np.sum(data['GEM5' ], axis=0),interpolation='none'); plt.gca().invert_yaxis(); plt.title('vrai GEM5'   )
                    plt.figure(2); plt.imshow(np.sum(data['COSPIN' ], axis=0),interpolation='none'); plt.gca().invert_yaxis(); plt.title('GEM5'   )
                    plt.figure(3); plt.imshow(np.sum(data['CALIPSO'], axis=0),interpolation='none'); plt.gca().invert_yaxis(); plt.title('CALIPSO')
                    plt.figure(4); plt.imshow(mask_bukovsky,interpolation='none');                   plt.gca().invert_yaxis(); plt.title('Mask Bukovsky')
                    plt.figure(5); plt.imshow(mask_track,interpolation='none');                      plt.gca().invert_yaxis(); plt.title('Mask track'   )
                    plt.figure(6); plt.imshow(ds_COSPIN['latitude' ],interpolation='none');          plt.gca().invert_yaxis(); plt.title('GEM5 latitude' )
                    plt.figure(7); plt.imshow(ds_COSPIN['longitude'],interpolation='none');          plt.gca().invert_yaxis(); plt.title('GEM5 longitude')
                    plt.show()
                    exit()


                profils_original  = {}
                profils_formatted = {}
                indices = np.where(mask == 1)
                N = len(indices[0]) 

                for dataset in datasets:
                    Nlevin = data[dataset].shape[0]
                    profils_original[dataset] = np.zeros([N, Nlevin]) * np.nan
                    if index == 0:

                        #column_sum           [dataset] = np.zeros(Nlevin)
                        #column_n             [dataset] = np.zeros(Nlevin)
                        column_formatted_sum [dataset] = np.zeros(Nlevout)
                        column_formatted_n   [dataset] = np.zeros(Nlevout)


                        if dataset == 'CALIPSO':
                            column_formatted_sum[dataset] = np.zeros(Nlevout)
                            column_formatted_n  [dataset] = np.zeros(Nlevout)

                    ### Extracting data to create profil: (2D x Nlevin --> N x Nlevin)
                    for n in range(N):
                        i = indices[0][n]
                        j = indices[1][n]
                        profils_original[dataset][n,:] = data[dataset][:,i,j]

                    ### Extracting layer height for COSPIN: (2D x Nlevin --> Nlevin X N)  oui on besoin de Nlevion x N à l'envers, il faudrait modifier le code pour tout uniformisier
                    if dataset == 'COSPIN':
                        layers[dataset] = np.zeros([Nlevin,N])
                        for n in range(N):
                            i = indices[0][n]
                            j = indices[1][n]
                            layers[dataset][:,n] = layers_COSPIN_2DxNlevin[:,i,j]
                        overlap_coeff[dataset] = compute_overlap_coeff(layers[dataset], layerout)


                    ### Formatting vertical levels: (N x Nlevin --> N x Nlevout)
                    if   dataset == layerout_dataset: profils_formatted[dataset] = profils_original[dataset]
                    else                            : profils_formatted[dataset] = format_levels(profils_original[dataset], overlap_coeff[dataset], Nlevout)


                    ### Aggregating the N individual profils into a single column (N x Nlevout --> 1 x Nlevout)
                    column_formatted_sum[dataset] +=  np.nansum( profils_formatted[dataset]                         , axis=0)
                    column_formatted_n  [dataset] +=  np.sum   ( np.where(np.isnan(profils_formatted[dataset]) ,0,1), axis=0)


                    #column_sum[dataset]  = column_sum[dataset] + np.nansum( profils_original[dataset]                         , axis=0)
                    #column_n  [dataset]  = column_n  [dataset] + np.sum   ( np.where(np.isnan(profils_original[dataset]) ,0,1), axis=0)




            create_output(column_formatted_sum, column_formatted_n, layerout, dirout,  bukovsky_region, YYYYMM)
                #if index == 2:
                    #plt.plot(np.flipud(column_formatted_sum['COSPIN' ] / column_formatted_n['COSPIN' ] ), layerout, 'g-',label="GEM5")
                    #plt.plot(np.flipud(column_formatted_sum['COSPOUT'] / column_formatted_n['COSPOUT'] ), layerout, 'b-',label="COSPOUT")
                    #plt.plot(np.flipud(column_formatted_sum['CALIPSO'] / column_formatted_n['CALIPSO'] ), layerout, 'r-',label="CALIPSO")
                    #plt.legend()
                    #plt.show()



    

if __name__ == '__main__':

    # Hardcoded
    format_level_method = 'weighted_mean'
    bukovsky_regions = ['ColdNEPacific', 'WarmNEPacific', 'Southwest', 'NPlains', 'GreatLakes','WarmNWAtlantic','EastBoreal', 'Hudson', 'PacificNW', 'Appalachia']
    datasets = [ 'COSPIN', 'COSPOUT', 'CALIPSO']
    layerout_dataset = 'COSPOUT'

    # Call main
    main()

