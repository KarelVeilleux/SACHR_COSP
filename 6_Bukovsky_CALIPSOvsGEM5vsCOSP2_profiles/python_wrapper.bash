#!/bin/bash


###########################################################################
#  Environnement                                                          #
###########################################################################
module load utils/cdo                           &> /dev/null
module load python3/miniconda3 python3/python3  &> /dev/null
source activate base_plus                       &> /dev/null


args="$@"

echo "$args"
# Extracting the script
script1=$(echo "$args" | sed -n "s/.*-s \([^ ]*\).*/\1/p")
script2=$(echo "$args" | sed -n "s/.*--script \([^ ]*\).*/\1/p")
[ -z $script1 ] && script=$script2 || script=$script1

# Launching the script
if [[ $script == *.py ]]; then
    echo "$script"
    python $script $args -cdo $(which cdo)
fi

#python $python_script -c $configuration -ds $dataset -n $nomvar -pw $password -u $user -cdo $(which cdo) 