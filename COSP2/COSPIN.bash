module load python3/miniconda3 python3/python3 #&> /dev/null
source activate base_plus                      #&> /dev/null






DATE0=2013010100


#YYYYMMDDi=20140101; YYYYMMDDf=20140132; soumet COSPIN.bash -args $YYYYMMDDi $YYYYMMDDf -jn COSPIN_${YYYYMMDDi}-${YYYYMMDDf}
#YYYYMMDDi=20140101; YYYYMMDDf=20140132;        COSPIN.bash       $YYYYMMDDi $YYYYMMDDf 


YYYYMMDDi=$1
YYYYMMDDf=$2


YYYY=${YYYYMMDDi:0:4}



DATEi=${YYYYMMDDi}0000;
DATEf=${YYYYMMDDf}2300




MP_CONFIG='MPB'
#files_list='/pampa/poitras/DATA/CAL_LID_L2_05kmCPro-Standard-V4/list/bc_coast_11_2014.txt'
#files_list='/pampa/poitras/DATA/CAL_LID_L2_05kmCPro-Standard-V4/list/bermuda_azores_11_2014.txt'
#files_list='/pampa/poitras/DATA/CAL_LID_L2_05kmCPro-Standard-V4/list/great_lakes_11_2014.txt'
#files_list='/pampa/poitras/DATA/CAL_LID_L2_05kmCPro-Standard-V4/list/hudson_bay_11_2014.txt'
#files_list='/pampa/poitras/DATA/CAL_LID_L2_05kmCPro-Standard-V4/list/pacific_sw_11_2014.txt'
#files_list=/pampa/poitras/DATA/CALIPSO/CAL_LID_L2_05kmCPro-Standard-V4/list/NAM11_${YYYY}.txt
files_list=/pampa/poitras/DATA/MODIS/MCD06_L2/list/NAM11_${YYYY}.txt
overwrite=0

#####################################################################################################################################################

gemname=COSP2_NAM-11m_ERA5_GEM5_CLASS_NEWVEG_newP3-SCPF_SN_Lakes
gempath=Cascades_CORDEX/CLASS/$gemname
diri0=/pampa/poitras/DATA/GEM5/COSP2/$gempath/Samples_NetCDF
dirix=/pampa/poitras/DATA/GEM5/COSP2/$gempath/Samples_NetCDF

ncfiles_pm0=${diri0}/${gemname}_step0/pm${DATE0}_00000000p.nc # FOR MG
ncfiles_dm0=${diri0}/${gemname}_step0/dm${DATE0}_00000000p.nc # FOR ME
ncfiles_pmx=${dirix}/${gemname}_YYYYMM/pm${DATE0}_YYYYMMDDd.nc
ncfiles_dmx=${dirix}/${gemname}_YYYYMM/dm${DATE0}_YYYYMMDDd.nc
cosp_input_dir=/pampa/poitras/DATA/COSP2/$gempath/$MP_CONFIG/INPUT/NAM11/$YYYY

[ -d $cosp_input_dir ] || mkdir -p $cosp_input_dir



######################################################################################################################################################
# GENERATE INPUT FILES
python /home/poitras/SCRIPTS/COSP2/cosp2_input_generate_main_devX.py $ncfiles_pm0 $ncfiles_dm0 $ncfiles_pmx $ncfiles_dmx $cosp_input_dir $MP_CONFIG $DATEi $DATEf $files_list $overwrite


