#!/bin/bash


LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sca/compilers_and_tools/netcdf-f/gfortran/lib
ncinfo=/sca/compilers_and_tools/python/miniconda3/envs/base_plus/bin/ncinfo


#########################################################################################################################
#                                                      Input parameters                                                 #
#########################################################################################################################
# Time parameters

#YYYY=2014
#MM=01
#di=01; 
#df=32
#instruments='modis'

#YYYYMMDDi=20140101; YYYYMMDDf=20140132; instruments=modis; soumet COSPOUT.bash -args $YYYYMMDDi $YYYYMMDDf $instruments -jn COSPOUT_${instruments}_${YYYYMMDDi}-${YYYYMMDDf}
#YYYYMMDDi=20140101; YYYYMMDDf=20140132; instruments=modis;        COSPOUT.bash       $YYYYMMDDi $YYYYMMDDf $instruments

YYYYMMDDi=$1
YYYYMMDDf=$2
instruments=$3 

if [[ "$instruments" == "calipso" ]]; then
	instruments='calipso_cloudprofile calipso_cloudprofile_phase calipso_cloudmap'
	echo need to be fixed
	exit
fi

DATEi=${YYYYMMDDi}0000; 
DATEf=${YYYYMMDDf}2300

#DATEi=201511010000; DATEf=201512320000
dtime=1H
DATE_length=${#DATEi}

#YYYYMMDD=${DATEi:0:7}
YYYYi=${DATEi:0:4}
YYYYf=${DATEf:0:4}

if [ $YYYYi -ne $YYYYf  ]; then
	echo "Date must be from the same year: $YYYYMMDDi --> $YYYYi, $YYYYMMDDf --> $YYYYf"
	echo exit
	exit
else
	YYYY=$YYYYi
fi





# COSP2 input/output settings
nsubcol=1
moment=1
dl=480



#instruments='atlid cloudsat grdlidar misr calipso isccp modis example'
#instruments='calipso_cloudprofile'
#instruments='calipso_cloudprofile_phase'
#instruments='calipso_cloudmap'
#instruments='calipso_cloudprofile calipso_cloudprofile_phase calipso_cloudmap'
#instruments='modis'
config_in_template=cosp2_input_template_001.txt

workdir=/pampa/poitras/SCRATCH/COSP2/workdir_$$


if   [ $dl -eq 480 ]; then cosprun_dir=/chinook/poitras/COSPv2.0/driver/run
elif [ $dl -eq  60 ]; then cosprun_dir=/chinook/poitras/COSPv2.0_60m/driver/run
fi
cosprun_dir=/pampa/poitras/SCRATCH/run
cp -rf $cosprun_dir $workdir




[ $moment -eq 2 ] && nmoment=MMF_v3.5_two_moment || nmoment=MMF_v3_single_moment 



# Input/output data
MP_CONFIG='MPB'
overwrite=0
domain=NAM11            ; npoints=429025
gempath=Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes
dir_cospin_data=/pampa/poitras/DATA/COSP2/$gempath/$MP_CONFIG/INPUT/$domain/$YYYY
dir_cospout_data=/pampa/poitras/DATA/COSP2/$gempath/$MP_CONFIG/OUTPUT/$domain/${moment}M$(printf "%03d" $nsubcol)SC/${instruments}_${dl}m



list=/pampa/poitras/DATA/CALIPSO/CAL_LID_L2_05kmCPro-Standard-V4/list/NAM11_${YYYY}.txt
DATES=$( cat $list | awk '{print $5}'); dates=( $DATES );
MMS=$(   cat $list | awk '{print $6}'); mms=(   $MMS   );
N=${#dates[@]};

if [[ "$instruments" == "modis" ]]; then
	list=/pampa/poitras/DATA/MODIS/MCD06_L2/list/NAM11_${YYYY}.txt
	DATES=$( cat $list | awk '{print $2}'); dates=( $DATES );
	MMS=$(   cat $list | awk '{print $3}'); mms=(   $MMS   );
	N=${#dates[@]};
fi





# Creating output directories (if not existing yet)
[ -d $dir_cospout_data ] || mkdir -p $dir_cospout_data
#[ -d $dir_cospout_fig  ] || mkdir -p $dir_cospout_fig


#########################################################################################################################
#							Running COSP							#
#########################################################################################################################
for instrument in $instruments; do
	echo $instrument
	DATE=$DATEi;
	while [ $DATE -le $DATEf ]; do
		in_the_list=$(echo $DATES | grep $DATE | wc -l)
		echo ${dir_cospin_data}/cosp_input_${DATE}.nc $in_the_list
		if [ $in_the_list -gt 0 ]; then
			cospin_file=${dir_cospin_data}/cosp_input_${DATE}.nc
			if [ -f $cospin_file ]; then

				#########################################################################################
				# Create a link to the input file (because the full path is too long to be used by COSP2)
				ln -sf $cospin_file $workdir/cosp_input.nc

				#########################################################################################
				# Create a link to the desired output configuration
				config_out=${workdir}/preconfig_output/cosp2_output_${instrument}.txt
				ln -sf $config_out ${workdir}/cosp2_output_nl.txt

				#########################################################################################
				# Copy and modified the input template
				template=${workdir}/input_template/$config_in_template
				config_in=${workdir}/cosp2_input_nl.txt
				[ -L $config_in ] && rm $config_in 
				cp $template $config_in
				cospout_file_1D=$dir_cospout_data/cospout_${DATE}.nc
				cospout_file_2D=$dir_cospout_data/cospout_${DATE}_2D.nc
				cospin_file=${workdir}/cosp_input.nc
               	 		sed  -i "s:xxxFOUTPUTxxx:${cospout_file_1D}:g" $config_in
				sed  -i "s:xxxNSUBCOLxxx:${nsubcol}:g"         $config_in
                		sed  -i "s:xxxNMOMENTxxx:${nmoment}:g"         $config_in
				sed  -i "s:xxxNPOINTSxxx:${npoints}:g"         $config_in
				sed  -i "s:xxxFINPUTxxx:${cospin_file}:g"      $config_in
				#########################################################################################
                		# Running COSP2


				#Check if the files exist and are readable (not corrupted)
		        	# stat=1 : Exist + not corrputed (fileo)
        			# stat=0 : Does not exist
        			# stat=-1: Exist + IS corrupted (fileo)
				filei=$cospin_file
				fileo1D=$cospout_file_1D
				fileo2D=$cospout_file_2D
        			[ -f $filei   ] && { $ncinfo $filei   &> /dev/null  &&   stati=1 ||   stati=-1; } ||   stati=0
				[ -f $fileo1D ] && { $ncinfo $fileo1D &> /dev/null  && stato1D=1 || stato1D=-1; } || stato1D=0
        			[ -f $fileo2D ] && { $ncinfo $fileo2D &> /dev/null  && stato2D=1 || stato2D=-1; } || stato2D=0

				if   [ $overwrite -eq  1 ] && [ $stati     -eq  1 ];  then  message="$fileo1D  was created (new/overwritten)"                        ; run_cosp=1;
				elif [ $overwrite -eq  1 ] && [ $stati     -ne  1 ];  then  message="$fileo1D  cannot be created. Input file is not readable: $filei"; run_cosp=0;
				elif [ $overwrite -eq  0 ] && [ $stato2D   -eq  1 ];  then  message="$fileo2D  already exists and is readable (skip)"                ; run_cosp=0;
				elif [ $overwrite -eq  0 ] && [ $stato1D   -eq  1 ];  then  message="$fileo1D  already exists and is readable (skip)"                ; run_cosp=0;
				elif [ $overwrite -eq  0 ] && [ $stati     -eq  1 ];  then  message="$fileo1D  was created (new)"                                    ; run_cosp=1;
				fi	
			
				if [ $run_cosp -eq 1 ]; then
					[ -f $fileo1D ]  && rm $fileo1D
					cd $workdir
					./cosp2_test  cosp2_input_nl.txt &> /dev/null
					[ -f $fileo1D ] && { $ncinfo $fileo1D &> /dev/null  && stato1D=1 || stato1D=-1; } || stato1D=0
					if [ $stato1D -ne  1 ]; then
						message="$fileo1D was not created correctly (will be deleted)"
						[ -f $fileo1D ] && rm $fileo1D
					fi	
				fi
				echo $message 
			fi
		fi
		
		#########################################################################################
		# Date increment
        	newDATE=$(r.date $DATE +$dtime)
        	DATE=${newDATE:0:$DATE_length}
	done
done

echo
#########################################################################################################################
#                                                  Converting output in 2D                                              #
#########################################################################################################################
# This part and the cosp part are separated because loading python module interfer with COSP environement
module load python3/miniconda3 python3/python3 #>& /dev/null
source activate base_plus                      #>& /dev/null


echo "Converting output 1D --> 2D"
for instrument in $instruments; do
        DATE=$DATEi;
        while [ $DATE -le $DATEf ]; do
		in_the_list=$(echo $DATES | grep $DATE | wc -l)
                if [ $in_the_list -gt 0 ]; then
			cospout_file=$dir_cospout_data/cospout_${DATE}.nc
			if [ -f $cospout_file ]; then
                		#########################################################################################################
				# Converting output file into output file 2D
				cospin_file=${dir_cospin_data}/cosp_input_${DATE}.nc
				cospout_file_2D=$dir_cospout_data/cospout_${DATE}_2D.nc
		
				#########################################################################################################
				# Converting output file into output file 2D
				#echo $cospout_file $cospout_file_2D $cospin_file
				python /home/poitras/SCRIPTS/COSP2/cosp2_output_format_2D.py $cospout_file $cospout_file_2D $cospin_file &> /dev/null
	        		[[ "$instrument" == "example" ]] || rm   $cospout_file
				echo $cospout_file_2D
                	fi
		fi
		#########################################################################################################
		# Date increment
                newDATE=$(r.date $DATE +$dtime)
                DATE=${newDATE:0:$DATE_length}
		
		
	        
        done
done

rm -rf $workdir











