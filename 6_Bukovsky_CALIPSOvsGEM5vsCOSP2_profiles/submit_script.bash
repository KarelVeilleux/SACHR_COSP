#!/bin/bash

#------------------------------------------------------------------------------
# Environment
#------------------------------------------------------------------------------
module load python3/miniconda3 python3/python3  &> /dev/null
source activate base_plus                       &> /dev/null



#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
function set_full_path() {
    args=$1
    key1=$2
    key2=$3
    path=$4
    if [[  "$args" =~ $key1|$key2 ]]; then
        full_path=$(readlink -f $path)
        args=$(echo "$args" | sed "s#\($key1\|$key2\) $path#\1 $full_path#g")
    fi  
}


function extract_value() {
    args=$1
    key1=$2
    key2=$3
    if [[  "$args" =~ $key1|$key2 ]]; then
        value1=$(echo "$args" | sed -n "s/.*$key1 \([^ ]*\).*/\1/p")
        value2=$(echo "$args" | sed -n "s/.*$key2 \([^ ]*\).*/\1/p")
        [ -z $value1 ] && value=$value2 || value=$value1
        echo $value
   fi
}



#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
args="$@"

# Sanity check
python sanity_check.py $args
[ $? -ne 0 ] && { echo "Sanity check failed. Exit"; exit; }


# Extract some values from the parsed arguments
script=$(       extract_value "$args" "-s"  "--script"       )
configuration=$(extract_value "$args" "-c"  "--configuration")
year=$(         extract_value "$args" "-y"  "--year"      )
month=$(        extract_value "$args" "-m"  "--month"       )

# Set full paths
set_full_path "$args" "-s" "--script"        $script
set_full_path "$args" "-c" "--configuration" $configuration


# Set jobname (for scheduler)
[ $month -lt 10 ] && m=0$month || m=$month
jn=${script}_${year}${m}


# Submission
python_wrapper=$(dirname $(readlink -f $0))/python_wrapper.bash
args_for_soumet=$(echo "$args" | sed -E "s/(-{1,2}[a-zA-Z0-9]+) ([^ ]+)/'\1 \2'/g")

soumet $python_wrapper -args "$args_for_soumet" -cpus 1 -t 864000  -listing $LISTINGS -jn $jn
#$python_wrapper "$args"


# Exemple de soumission pour le mois d'avril 2014 avec les directory dans config.yml :
# ./submit_script.bash -s main_CALIPSO_vs_GEMCOSP_01_compute_profile.py -c ../config.yml  -m 4 -y 2014

