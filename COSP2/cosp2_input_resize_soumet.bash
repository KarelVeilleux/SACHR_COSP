module load python3/miniconda3 python3/python3 >& /dev/null
source activate base_plus                      >& /dev/null



YYYY=2014
MP_CONFIG='MPB'
domain_original=NAM11
dir_input_cosp=/pampa/poitras/DATA/COSP2/Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/$MP_CONFIG/INPUT
dir_list=/pampa/poitras/DATA/CAL_LID_L2_05kmCPro-Standard-V4/list


   


overwrite=0
#domain_modified=box11            ; lati=205; latf=490; loni=250; lonf=575; levi=0; levf=62
domain_modified=bc_coast_11      ; xi=060; xf=190; yi=355; yf=480; levi=0; levf=62
domain_modified=bermuda_azores_11; xi=530; xf=615; yi=145; yf=230; levi=0; levf=62  #Bermuda-Azores High
#domain_modified=great_lakes_11   ; xi=315; xf=430; yi=262; yf=338;   #Great Lakes
#domain_modified=hudson_bay_11    ; xi=290; xf=410; yi=355; yf=480;  #Hudson bay
#domain_modified=pacific_sw_11    ; xi=040; xf=100; yi=100; yf=200;  #Pacific-SW
#domain_modified=sonora_desert_11 ; xi=130; xf=185; yi=140; yf=230;   #Sonara desert


dir_original=$dir_input_cosp/$domain_original/$YYYY
dir_modified=$dir_input_cosp/$domain_modified/$YYYY
list=$dir_list/${domain_modified}_${YYYY}.txt

[ -f $lst          ] || { echo $list does not exit && exit; }
[ -d $dir_modified ] || mkdir -p $dir_modified


DATE=$( cat $list |  awk '{print $5}'); date=( $DATE );
N=${#date[@]};
n=0; while [ $n -lt  $N ]; do

        d=${date[$n]}
	filei=$dir_original/cosp_input_${d}.nc
        fileo=$dir_modified/cosp_input_${d}.nc

	#Check if the files exist and are readable (not corrupted)
        # stat=1 : Exist (filei);   Exist + not corrputed (fileo)
        # stat=0 : Does not exist
        # stat=-1: Exist + IS corrupted (fileo)
        [ -f $filei ] && {                                stati=1            ; } || stati=0
        [ -f $fileo ] && { ncinfo $fileo &> /dev/null  && stato=1 || stato=-1; } || stato=0

        if   [ $stato -eq  1 ] && ! [ $overwrite -eq 1 ]; then message="$fileo  already exists and is  readable (skip)"              ; perform_resizing=0;
        elif [ $stato -eq  1 ] &&   [ $overwrite -eq 1 ]; then message="$fileo  already exists and was readable (overwrite)"         ; perform_resizing=1;
        elif [ $stato -eq -1 ] &&   [ $stati     -eq 1 ]; then message="$fileo  already exists but is  CORRUPTED (overwrite)"        ; perform_resizing=1;
        elif [ $stato -eq  0 ] &&   [ $stati     -eq 1 ]; then message="$fileo  has been created"                                    ; perform_resizing=1;
        elif [ $stato -eq  0 ] &&   [ $stati     -eq 0 ]; then message="$fileo  cannot be created. Input file does not exist: $filei"; perform_resizing=0;
	fi

	# Performin the actual resizing
        if   [ $perform_resizing -eq  1 ]; then
        
                [ -f $fileo ] &&  rm $fileo                               
		python /home/poitras/SCRIPTS/COSP2/cosp2_input_resize.py  $yi $yf $xi $xf $levi $levf $filei $fileo #&> /dev/null
                
		# Check if the resizing have been performed correctly
                [ -f $fileo ] && { ncinfo $fileo &> /dev/null  && stato=1 || stato=-1; } || stato=0
                if [ $stato -ne  1 ]; then
                        message="$fileo  was not created correctly (file will be deleted)"
                        rm $fileo
                fi
        fi
        echo $message

	



#	if [  -f $file_modified ]; then
#		ncinfo $file_modified &> /dev/null && is_ok=1 || is_ok=0
#
#		if [ $is_ok -eq 1 ]; then
#			if  [ $overwrite -ne 1 ]; then
#				echo $n/$N $file_modified  'is ok'
#			else
#				echo $n/$N $file_modified  'is ok (will be overwritten)'
#
#		else
#			 echo $n/$N $file_modified  'is corrupted (will be overwritten)'
#			 rm $file_modified
#		fi
#
#	else
#		if [  -f $file_original ]; then
#			python /home/poitras/SCRIPTS/COSP2/cosp2_input_resize.py  $yi $yf $xi $xf $levi $levf $file_original $file_modified &> /dev/null && is_ok=1 || is_ok=0
#			if [ $is_ok -eq 1 ]; then 
#			      	echo $n/$N $file_modified created
#			else
#				echo $n/$N $file_modified 'was not created correctly (will be removed)'
#				[ -f $file_modified ] && rm $file_modified
#			fi
#		else
#			echo $n/$N $file_original  not found
#		fi
#
#	fi

	n=$((n+1))
done






















exit
module unload python3/python-rpn                  &> /dev/null
module unload python3/miniconda3 python3/python3  &> /dev/null
module load utils/cdo

n=0; while [ $n -lt  $N ]; do
        d=${date[$n]}
        
        file_modified=$dir_modified/cosp_input_${d}.nc
        if [   -f $file_modified ]; then
		
		ncdump $file_modified | grep ncdump | grep error
	fi
        n=$((n+1))
done



ncdump /pampa/poitras/DATA/COSP2/Cascades_CORDEX/CLASS/COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes/MPB/INPUT/great_lakes_11/2014/cosp_input_201412161900.nc




exit
[ -d $dir_modified ] || mkdir -p $dir_modified



######################################################################################################################################################
# GENERATE INPUT FILES
#echo "=== GENERATE INPUT FILES ==="
#python /home/poitras/SCRIPTS/COSP2/cosp2_input_generate_main_dev.py $ncfiles_pm0 $ncfiles_dm0 $ncfiles_pmx $ncfiles_dmx $cosp_input_dir $MP_CONFIG $YYYYMM



# RESIZE INPUT FILES
echo "=== RESIZE INPUT FILES (+ GENERATING FIGURES) ==="
DATE=$DATEi;
while [ $DATE -le $DATEf ]; do
	
	cosp_input_file_original_size=${cosp_input_dir}/cosp_input_${DATE}.nc
	cosp_input_file_resized=${cosp_input_dir}/RESIZED/cosp_input_${DATE}.nc
	
	if [ ! -f $cosp_input_file_original_size ]; then
		#echo $cosp_input_file_original_size does not exist;
		#exit
		zzz=1
	else
		echo $cosp_input_file_original_size
		python /home/poitras/SCRIPTS/COSP2/cosp2_input_resize.py  $lati $latf $loni $lonf $levi $levf $cosp_input_file_original_size $cosp_input_file_resized
		#python /home/poitras/SCRIPTS/COSP2/cosp2_input_plot_2D.py $lati $latf $loni $lonf $ncfile_cloudsat $cosp_input_file_resized $cospin_fig_dir
		#python /home/poitras/SCRIPTS/COSP2/cosp2_input_plot_3D.py $lati $latf $loni $lonf $ncfile_cloudsat $cosp_input_file_resized $cospin_fig_dir
	fi
	newDATE=$(r.date $DATE +$dtime)
	DATE=${newDATE:0:$DATE_length}
done


#mv    ${cosp_input_dir}/RESIZED/cosp_input_${YYYYMM}??.nc ${cosp_input_dir}/.
#rmdir ${cosp_input_dir}/RESIZED

