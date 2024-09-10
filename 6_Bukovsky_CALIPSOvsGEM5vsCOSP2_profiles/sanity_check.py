#!/usr/bin/env python3

__author__     = "Vincent Poitras"
__credits__    = "..."
__maintainer__ = "Vincent Poitras"
__email__      = "vincent.poitras@ec.gc.ca"
__status__     = "Development"


import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auxilliary_functions import get_args

def main():

    # Get args 1/2 (here, only to get script)
    args = get_args(sys.argv) 
    script = args['script']

    # Check if argument script is missing
    if script is None:
        print('[ERROR] Argument script is missing')
        print('[ERROR] ... --script name_of_your_script')
        print('[ERROR] Exit')
        exit(1)

    # Check if the script path exists
    if not os.path.exists(script):
        print('[ERROR] %s does not exist' % script)
        print('[ERROR] Exit')
        exit(1)
    
    # Get args 2/2 (here, specifying the script in order to have an error message if a requiered arg is missing)
    args = get_args(sys.argv, script)
    for arg, value in args.items():
        globals()[arg] = value

    # Check if configuration path exist (if requiered)
    if configuration is not None:
        if not os.path.exists(configuration):
            print('[ERROR] %s does not exist' % configuration)
            print('[ERROR] Exit')
            exit(1)

    # Check on year
    yrange = [2014,2015]
    if year is not None:
        if (year < yrange[0]) or (year > yrange[1]):
            print('[ERROR] year = %d is not in the range %s ' % (year, yrange))
            print('[ERROR] Exit')
            exit(1)

    # Check on month
    mrange = [1,12]
    if month is not None:
        if (month < mrange[0]) or (month > mrange[1]):
            print('[ERROR] year = %d is not in the range %s ' % (month, mrange))
            print('[ERROR] Exit')
            exit(1)



    # Check on dataset*
    none_flag= False
    ds_allowed_values = ['RDRS', 'ERA5', 'MERRA2']
    pattern = r"dataset\d+" 
    for arg in args:
        if bool(re.match(pattern,arg)) or arg == 'dataset':
            if args[arg] is not None:
                if args[arg] not in ds_allowed_values:
                    print('[ERROR] %s = %s' % (arg,args[arg]))
                    print('[ERROR] Allowed values are %s:' % ds_allowed_values)
                    print('[ERROR] Exit')
                    exit(1)


if __name__ == "__main__":
    main()

