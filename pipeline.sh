#!/bin/bash

MS=$1
BF=$2
datetime=$3
dirroot=/mnt/LOFAR-PSP/pearse_2ndperihelion/MS/images
datadir=$(echo "${datetime::10}" | tr '-' '_')
mkdir "$dirroot"/"$datadir"

#starting from raw MS dataset and BF data, find time of max radio burst, calibrate MS and make radio image.

#Calibrate
python update_parset.py "$MS" -t "$datetime"
MS_av=${MS::-3}_aw_av.MS
#Find burst time
cd /mnt/LOFAR-PSP/pearse_2ndperihelion/BF || exit
python find_maxburst.py "$BF" "$datetime"
bursttime=$(tail -n 1 peak_times.txt)
cd - || exit
interval=$(python time_to_wsclean_interval.py "$MS_av" -t "$bursttime")
IFS=' '
read -ra intarr <<< "$interval"
#WSClean
wsclean -mem 85 -no-reorder -no-update-model-required -mgain 0.8 -weight briggs 0 -size 1024 1024 -scale 5asec -pol I \
        -data-column CORRECTED_DATA -auto-mask 2 -auto-threshold 0.3 \
        -multiscale -multiscale-scale-bias 0.7 -multiscale-scales 90,135,180,225,270,315,360,405,450 \
        -niter 10000 -fit-beam -intervals-out 1 -interval "${intarr[1]}" "${intarr[2]}" \
        -name "$datadir"/