module load python3/miniconda3 python3/python3;
source activate base_plus;




# TIMESTEP TO TREAT
YYYYMMDD=20140105; DATEi=201401050100; DATEf=201401060000
#YYYYMMDD=20140908; DATEi=201409080100; DATEf=201409090000
#YYYYMMDD=20140909; DATEi=201409090100; DATEf=201409100000
#YYYYMMDD=20140910; DATEi=201409100100; DATEf=201409110000
#YYYYMMDD=20140911; DATEi=201409110100; DATEf=201409120000
#YYYYMMDD=20140915; DATEi=201409150100; DATEf=201409160000
YYYYMMDD=20140105; DATEi=201401051900; DATEf=201401051900


YYYYMM=201401
#DATEi=${YYYYMM}010000; DATEf=${YYYYMM}320000;

dtime=1H
DATE_length=${#DATEi}


MP_CONFIG='MPB'
gemname=COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes
gempath=Cascades_CORDEX/CLASS/$gemname
diri0=/pampa/poitras/DATA/TREATED/GEM5/COSP2/$gempath/Samples_NetCDF
dirix=/pampa/poitras/DATA/TREATED/GEM5/COSP2/$gempath/Samples_NetCDF
ncfiles_pm0=${diri0}/${gemname}_step0/pm2013010100_00000000p.nc # FOR MG
ncfiles_dm0=${diri0}/${gemname}_step0/dm2013010100_00000000p.nc # FOR ME
#ncfiles_pmx=${dirix}/${gemname}_201409/pm2013010100_${YYYYMMDD}d.nc
#ncfiles_dmx=${dirix}/${gemname}_201409/dm2013010100_${YYYYMMDD}d.nc

ncfiles_pmx=${dirix}/${gemname}_${YYYYMM}/pm2013010100_${YYYYMMDD}d.nc
ncfiles_dmx=${dirix}/${gemname}_${YYYYMM}/dm2013010100_${YYYYMMDD}d.nc
cosp_input_dir=/pampa/poitras/DATA/TREATED/COSP2/$gempath/$MP_CONFIG/INPUT/TEST

[ -d $cosp_input_dir ] || mkdir -p $cosp_input_dir



# TO RESIZE INPUT
lati=205
latf=490
loni=250
lonf=575
levi=0
levf=62
[ -d $cosp_input_dir/RESIZED ] || mkdir -p $cosp_input_dir/RESIZED


# TO CREATE INPUT FIGURES
ncfile_cloudsat=/pampa/poitras/DATA/ORIGINAL/CLOUDSAT/NetCDF/CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00/20140105180611_40917_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.nc
cospin_fig_dir=/pampa/poitras/figures/COSP2/$gempath/$MP_CONFIG/INPUT 
[ -d $cospin_fig_dir ] || mkdir -p $cospin_fig_dir

######################################################################################################################################################
# GENERATE INPUT FILES
echo "=== GENERATE INPUT FILES ==="
python /home/poitras/SCRIPTS/COSP2/cosp2_input_generate_main.py $ncfiles_pm0 $ncfiles_dm0 $ncfiles_pmx $ncfiles_dmx $cosp_input_dir $MP_CONFIG



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


#mv    ${cosp_input_dir}/RESIZED/*${YYYYMM} ${cosp_input_dir}/.
#rmdir ${cosp_input_dir}/RESIZED

