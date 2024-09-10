#!/bin/bash

# info: veilleux.karel@uqam.ca
# date: 2024-01
# description: "wrapper" pour lancer les script main_MODIS_vs_CALIPSO_01_compute_LowMidHigh_cloudcover_CALIPSO.py
#              C'est plus simple de soumettre un "wrapper" à travers l'ordonnanceur que le script python directement

###########################################################################
#                               Environnement                             #
###########################################################################
module load python3/miniconda3 python3/python3  &> /dev/null
source activate base_plus                       &> /dev/null

###########################################################################
#  Input parameters                                                       #
###########################################################################
working_directory=$1
YYYYMM=$2
layerdef=$3
overwrite=$4

###########################################################################
# Script launch                                                          #
###########################################################################
script=$working_directory/main_01_CALIPSOvsGEMvsCOSP_compute_LowMidHigh.py
args="$working_directory $YYYYMM $layerdef $overwrite"
python $script $args