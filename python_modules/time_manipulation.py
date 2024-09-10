from datetime import datetime, timedelta
import numpy as np
NaN = np.nan

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))



def convert_array_into_strptime(array,format,dim=1):
    if dim == 1:
        N =len(array)
        output = {}
        for n in range(N):
            output[n] = datetime.strptime( str(array[n]),format)
    return output
