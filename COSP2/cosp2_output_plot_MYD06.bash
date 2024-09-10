#!/bin/bash


module load python3/miniconda3 python3/python3 &> /dev/null
source activate base_plus                      &> /dev/null

####################################################################
script_python=/home/poitras/SCRIPTS/COSP2/cosp2_output_plot_MYD06.py
filelist=/pampa/poitras/DATA/MODIS/MYD06_L2/list/NAM11_2014.txt

python $script_python $filelist
