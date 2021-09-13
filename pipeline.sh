#!/bin/bash

MS=$1
BF=$2
datetime_start=$3
datetime_end=$4
dirroot=/mnt/LOFAR-PSP/pearse_2ndperihelion/MS/images
datadir=$(echo "${datetime_start::10}" | tr '-' '_')
mkdir -p "$dirroot"/"$datadir"
logfile="$dirroot"/"$datadir"/pipeline.log
touch "$logfile"
echo "Pipeline run $(date +%Y-%m-%dT%H:%M:%S)" 2>&1 | tee -a "$logfile"

#starting from raw MS dataset and BF data, find time of max radio burst, calibrate MS and make radio image.

#Calibrate
#python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/update_parset.py "$MS" -t "$datetime_start" "$datetime_end" 2>&1 | tee -a "$logfile"
MS_av=${MS::-3}_aw_av.MS/
###Find burst time
##cd /mnt/LOFAR-PSP/pearse_2ndperihelion/BF || exit 2>&1 | tee -a "$logfile"
##python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/find_maxburst.py "$BF" -t "$datetime_start" "$datetime_end" 2>&1 | tee -a "$logfile"
#
##Make images and fit in image space
#python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/run_wsclean.py "$MS_av" -n 320 -p /mnt/LOFAR-PSP/pearse_2ndperihelion/BF/peak_times_30MHz_"${datetime_start::10}"T120000_130000.pkl -o "$dirroot"/"$datadir"/30MHz/uncal/
#python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/plot_fits.py all -g -a "$dirroot"/"$datadir"/30MHz/uncal/\*image.fits
for i in $(ls "$dirroot"/"$datadir"/30MHz/*image.fits); do nohup python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/fit_gauss_image.py $i & done
wait
#Fit in visibility space
python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/plot_vis.py "$MS_av" -p /mnt/LOFAR-PSP/pearse_2ndperihelion/BF/peak_times_30MHz_"${datetime_start::10}"T120000_130000.pkl
#MS_av=${MS::-3}_aw_av.MS
##Find burst time
#cd /mnt/LOFAR-PSP/pearse_2ndperihelion/BF || exit 2>&1 | tee -a "$logfile"
#python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/find_maxburst.py "$BF" -t "$datetime_start" "$datetime_end" 2>&1 | tee -a "$logfile"
#bursttime=$(tail -n 1 peak_times.txt)
#cd - || exit 2>&1 | tee -a "$logfile"
#interval=$(python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/time_to_wsclean_interval.py "$MS_av" --trange "$bursttime")
#IFS=' '
#read -ra intarr <<< "$interval"
#echo "${intarr[*]}"
#WSClean
#if [ ! -f "$dirroot"/"$datadir"/wsclean-image.fits ]; then
#  wsclean -mem 85 -no-reorder -no-update-model-required -mgain 0.7 -weight briggs 0 -size 1024 1024 -scale 5asec -pol I \
#  -data-column CORRECTED_DATA -taper-gaussian 90 \
#  -multiscale -multiscale-scales 90,135,180,225,270,315,360,405,450 \
#  -niter 320 -fit-beam -intervals-out 1 -interval "${intarr[1]}" "${intarr[2]}" \
#  -name "$dirroot"/"$datadir"/wsclean "$MS_av" 2>&1 | tee -a "$logfile"
#
#  python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/fit_gauss_image.py "$dirroot"/"$datadir"/wsclean-image.fits 2>&1 | tee -a "$logfile"
#  python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/radio_euv_overlay.py "$dirroot"/"$datadir"/wsclean-image.fits 2>&1 | tee -a "$logfile"
#  python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/manual_clean.py "$dirroot"/"$datadir"/wsclean-model.fits 2>&1 | tee -a "$logfile"
#fi

#wsclean -mem 85 -no-reorder -no-update-model-required -mgain 0.6 -weight briggs 0 -size 1024 1024 -scale 5asec -pol I \
#-data-column CORRECTED_DATA -taper-gaussian 90 \
#-multiscale \
#-auto-threshold 1 -niter 2000 -fit-beam -intervals-out 1 -interval "${intarr[1]}" "${intarr[2]}" \
#-name "$dirroot"/"$datadir"/wsclean_auto "$MS_av" 2>&1 | tee -a "$logfile"
#
#python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/fit_gauss_image.py "$dirroot"/"$datadir"/wsclean_auto-image.fits 2>&1 | tee -a "$logfile"
#python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/radio_euv_overlay.py "$dirroot"/"$datadir"/wsclean_auto-image.fits 2>&1 | tee -a "$logfile"
#python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/manual_clean.py "$dirroot"/"$datadir"/wsclean_auto-model.fits 2>&1 | tee -a "$logfile"

#wsclean -mem 85 -no-reorder -no-update-model-required -mgain 0.6 -weight briggs 0 -size 1024 1024 -scale 5asec -pol I \
#-data-column CORRECTED_DATA -taper-gaussian 90 \
#-multiscale \
#-niter 320 -fit-beam -intervals-out 360  \
#-name "$dirroot"/"$datadir"/every5s/wsclean_auto_clock "$MS_av" 2>&1 | tee -a "$logfile"
#python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/run_wsclean.py
#for i in $(seq -w 1 360)
#do
##  python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/fit_gauss_image.py "$dirroot"/"$datadir"/every5s/wsclean_auto-t0"$i"-image.fits &
#  python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/radio_euv_overlay.py "$dirroot"/"$datadir"/every5s/wsclean_auto-t0"$i"-image.fits &
#  python /mnt/LOFAR-PSP/pearse_2ndperihelion/scripts/manual_clean.py "$dirroot"/"$datadir"/every5s/wsclean_auto-t0"$i"-model.fits &
#done