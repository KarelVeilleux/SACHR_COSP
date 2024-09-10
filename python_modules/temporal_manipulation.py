from datetime               import datetime
from dateutil.relativedelta import relativedelta
import numpy  as np
NaN = np.nan


#################################################################################################################################################################i
def compute_seasmean(input,initial_date, timestep):

    output = {}

    seas = {}
    seas['DJF'] = [1, 2,12];
    seas['MAM'] = [3, 4, 5];
    seas['JJA'] = [6, 7, 8];
    seas['SON'] = [9,10,11];

    #Convert initial date into a date object
    dtObj = datetime.strptime(initial_date, '%Y/%m/%d')

    # Assign a season to each timestep
    values_view    = input.values()
    value_iterator = iter(values_view)
    first_value    = next(value_iterator)
    nt             = len(first_value)
    season         =  ["" for t in range(nt)]


    for t in range(nt):
    
        if   timestep == 'hour' :  date = dtObj + relativedelta(hours  = t)
        elif timestep == 'day'  :  date = dtObj + relativedelta(days   = t)
        elif timestep == 'month':  date = dtObj + relativedelta(months = t)
        month = int(date.strftime('%m'))
        for s in seas:
            if month in seas[s]:
                season[t] = s
        #print(nt,t, date, month, season[t])

    # Computing seasonal mean
    for var in input:
        mean = {}
        for s in seas:
            data = input[var].copy()

            for t in range(nt):
                if season[t] != s:
                    data[t,:,:] = NaN
            mean[s] = np.nanmean(data,axis=(0))
        mean['ANNUAL'] = np.nanmean(input[var],axis=(0))
        output[var] = mean
    return output




#################################################################################################################################################################
def compute_multiyear_seasmean(input,season_order):
    
    output = {}

    # Assign a season to each timestep
    values_view    = input.values()
    value_iterator = iter(values_view)
    first_value    = next(value_iterator)
    nt             = len(first_value)
    season         =  ["" for t in range(nt)]
    n_season_order = len(season_order)

    for t in range(nt):
        season[t] = season_order [t % n_season_order]
        #print(t,season[t])
    
    # Computing seasonal mean
    for var in input:
        mean = {}
        for s in season_order:
            data = input[var].copy()

            for t in range(nt):
                if season[t] != s:
                    data[t] = NaN
            mean[s] = np.nanmean(data,axis=(0))
        output[var] = mean
    return output







#################################################################################################################################################################i
def compute_monthlymean(input, initial_date, timestep):
    
    output = {}

    #Convert initial date into a date object
    dtObj = datetime.strptime(initial_date, '%Y/%m/%d')
    
    # Assign a month and year to each timestep
    values_view    = input.values()
    value_iterator = iter(values_view)
    first_value    = next(value_iterator)
    nt             = len(first_value)
    month          = np.empty(nt,dtype=int)
    year           = np.empty(nt,dtype=int)
    yyyymm         = np.empty(nt,dtype=int)

    for t in range(nt):
        if   timestep == 'hour':  date = dtObj + relativedelta(hours = t)
        elif timestep == 'day' :  date = dtObj + relativedelta(days  = t)
        month [t] = int(date.strftime('%m'))
        year  [t] = int(date.strftime('%Y'))
        yyyymm[t] = year[t]*100 + month[t]  
        #print(nt,t, date, month[t],year [t])

    # Computing monthly mean
    ntOUT = len(set(yyyymm))
    for var in input:
        #nt   =  len(set(year))*12
        nx   =  input[var].shape[1]
        ny   =  input[var].shape[2]

        mean = np.empty((ntOUT,nx,ny))
        for yyyy in set(year):
            for mm in range(1,12+1):
                data = input[var].copy()

                for t in range(nt):
                    if  month[t] != mm  or year[t] != yyyy:
                        data[t] = NaN
                index = (yyyy-year[0])*12+mm-1
                #print(var,yyyy,mm,index)
                mean[index,:,:] = np.nanmean(data,axis=(0))
        output[var] = mean
    return output



#################################################################################################################################################################
def split_data_by_season(input,initial_date,timestep):

    output = {}

    #Convert initial date into a date object
    dtObj = datetime.strptime(initial_date, '%Y/%m/%d')

    month2season = {}
    month2season[ 1] = 'DJF'
    month2season[ 2] = 'DJF'
    month2season[ 3] = 'MAM'
    month2season[ 4] = 'MAM'
    month2season[ 5] = 'MAM'
    month2season[ 6] = 'JJA'
    month2season[ 7] = 'JJA'
    month2season[ 8] = 'JJA'
    month2season[ 9] = 'SON'
    month2season[10] = 'SON'
    month2season[11] = 'SON'
    month2season[12] = 'DJF'

    for var in input:
        nt   =  input[var].shape[0]
        data = {}
        for t in range(nt):
            if   timestep == 'hour' :  date = dtObj + relativedelta(hours  = t)
            elif timestep == 'day'  :  date = dtObj + relativedelta(days   = t)
            elif timestep == 'month':  date = dtObj + relativedelta(months = t)
            month          = int(date.strftime('%m'))
            season         = month2season[month]
            data_timeslice = np.expand_dims(input[var][t,:,:], axis=0)
            if season in data:
                data[season] = np.append(  data[season], data_timeslice, axis=0 )
            else:
                data[season] = data_timeslice
        data['ANNUAL'] = input[var]
        output[var] = data
    return output

