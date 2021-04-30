#!/bin/bash
eval "$(pyenv init -)" 
#DPPP cal_autoweight.parset
#DPPP sun_autoweight.parset
#DPPP sun_cal.parset
#DPPP sun_flag.parset
pyenv shell system
PYTHONPATH=/opt/lofarsoft/lib/python2.7/site-packages
echo "running fix_cal.py $1/instrument" 
python ../scripts/fix_cal.py $1/instrument
pyenv shell paper2
PYTHONPATH=
pyenv version
DPPP sun_applycal.parset
DPPP sun_applybeam.parset
