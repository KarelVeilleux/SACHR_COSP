#!/bin/bash
module load python3/miniconda3 python3/python3;
source activate base_plus;

python /home/poitras/SCRIPTS/COSP2/cosp2_output_plot_cloudprofile_phase.py

echo COMPLETED
