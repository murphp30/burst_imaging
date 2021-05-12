#!/usr/bin/env python

import argparse
import os

from sunpy.time import TimeRange

parser = argparse.ArgumentParser(description='creates parset files necessary \
                                 for solar radio imaging with LOFAR')

parser.add_argument('msin', help='input name of Measurement Set.', metavar='MSIN')
parser.add_argument('-t', '--trange', help='Time range over which to perform \
                    calibration. 2 arguments, must be of format YYYY-MM-DDTHH:MM:SS',
                    nargs=2, metavar=('START', 'END'))
parser.add_argument('-f', '--flag', help='run flag step. Default=False', action='store_true')
parser.add_argument('-d', '--dummy', help='doesn\'t run DPPP. Default=False', action='store_true')

args = parser.parse_args()
msin = args.msin
trange = args.trange
flag = args.flag
dummy = args.dummy

if trange is not None:
    trange = TimeRange(trange[0], trange[1])
    starttime = trange.start.datetime.strftime("%d%b%Y/%H:%M:%S.%f")
    endtime = trange.end.datetime.strftime("%d%b%Y/%H:%M:%S.%f")
    starttime = starttime[:-3]
    endtime = endtime[:-3]
elif trange is None:
    starttime = ' '
    endtime = ' '

if msin.split('SAP')[1][:3] == '001':
    cal_ms = msin
    sun_ms = msin.replace('SAP001', 'SAP000')
    sb_cal = int(cal_ms.split('SB')[1][:3])
    sb_sun = 'SB' + str(sb_cal - 60).zfill(3)
    sb_cal = cal_ms.split('_')[-2]
    sun_ms = sun_ms.replace(sb_cal, sb_sun)

elif msin.split('SAP')[1][:3] == '000':
    sun_ms = msin
    cal_ms = msin.replace('SAP000', 'SAP001')
    sb_sun = int(sun_ms.split('SB')[1][:3])
    sb_cal = 'SB' + str(sb_sun + 60).zfill(3)
    sb_sun = cal_ms.split('_')[-2]
    cal_ms = cal_ms.replace(sb_sun, sb_cal)

else:
    print("unexpected MS filename structure")

cal_aw = cal_ms.replace('_uv', '_uv_aw_avaoflag')
sun_aw = sun_ms.replace('_uv', '_uv_aw_av')

# with open('cal_autoweight.parset', 'w') as f:
#     f.write('msin={} \
#             \nmsout={} \
#             \nsteps=[] \
#             \nmsin.autoweight=True \
#             \nmsin.startchan = nchan/16 \
#             \nmsin.nchan = nchan*14/16\n'.format(cal_ms, cal_aw))

# os.system('echo Running autoweight on calibrator')
# if not dummy:# # 
#     os.system('DPPP cal_autoweight.parset')

with open('cal_aoflag.parset', 'w') as f:
    f.write('msin={} \
            \nmsout={} \
            \nmsin.starttime={} \
            \nmsin.endtime={} \
            \nmsin.autoweight=True \
            \nmsin.startchan = nchan/16 \
            \nmsin.nchan = nchan*14/16 \
            \nsteps=[flag, avg] \
            \nflag.type=aoflagger \
            \navg.type=average \
            \navg.timestep=5 \
            \navg.freqstep=16\n'.format(cal_ms, cal_aw, starttime, endtime))

os.system('echo Running autoweight, average and aoflag on calibrator')
if not dummy:
    os.system('DPPP cal_aoflag.parset')


# with open('sun_autoweight.parset', 'w') as f:
#     f.write('msin={} \
#             \nmsout={} \
#             \nsteps=[] \
#             \nmsin.autoweight=True \
#             \nmsin.startchan = nchan/16 \
#             \nmsin.nchan = nchan*14/16\n'.format(sun_ms, sun_aw))

# os.system('echo Running autoweight on sun')
# if not dummy:# # 
#     os.system('DPPP sun_autoweight.parset')

with open('sun_avg.parset', 'w') as f:
    f.write('msin={} \
            \nmsout={} \
            \nmsin.starttime={} \
            \nmsin.endtime={} \
            \nmsin.autoweight=True \
            \nmsin.startchan = nchan/16 \
            \nmsin.nchan = nchan*14/16 \
            \nsteps=[avg] \
            \navg.type=average \
            \navg.timestep=5 \
            \navg.freqstep=16\n'.format(sun_ms, sun_aw, starttime, endtime))

os.system('echo Running autoweight and average on sun')
if not dummy:
    os.system('DPPP sun_avg.parset')

with open('sun_cal.parset', 'w') as f:
    f.write('msin={} \
            \nmsout=. \
            \nmsin.starttime={} \
            \nmsin.endtime={} \
            \nsteps=[gaincal] \
            \ngaincal.usebeammodel=True \
            \ngaincal.solint=4 \
            \n#gaincal.timeslotsperparmupdate=100 \
            \ngaincal.sources=TauA \
            \ngaincal.sourcedb=TauA.sourcedb \
            \ngaincal.onebeamperpatch=True \
            \ngaincal.caltype=diagonal\n'.format(cal_aw, starttime, endtime))

os.system('echo calibrating data')
if not dummy:
    os.system('DPPP sun_cal.parset')

if flag:
    with open('sun_flag.parset', 'w') as f:
        f.write('msin={} \
                \nmsout=. \
                \nmsin.starttime={} \
                \nmsin.endtime={} \
                \nmsin.datacolumn=DATA \
                \nsteps=[flag] \
                \nflag.type=preflagger \
                \nflag.baseline=RS*&&* \
                \n#flag.mode=clear \n'.format(sun_aw, starttime, endtime))

    os.system('echo flagging baselines')
    if not dummy:
        os.system('DPPP sun_flag.parset')

with open('sun_applycal.parset', 'w') as f:
    f.write('msin={} \
            \nmsout=. \
            \nmsin.starttime={} \
            \nmsin.endtime={} \
            \nmsin.datacolumn=DATA \
            \nmsout.datacolumn=CORR_NO_BEAM \
            \nsteps=[applycal] \
            \napplycal.parmdb={}/instrument/ \
            \napplycal.updateweights=True\n'.format(sun_aw, starttime, endtime, cal_aw))

os.system('echo applying calibration')
if not dummy:
    os.system('DPPP sun_applycal.parset')

with open('sun_applybeam.parset', 'w') as f:
    f.write('msin={} \
            \nmsout=. \
            \nmsin.starttime={} \
            \nmsin.endtime={} \
            \nmsin.datacolumn=CORR_NO_BEAM \
            \nmsout.datacolumn=CORRECTED_DATA \
            \nsteps=[applybeam] \
            \napplybeam.updateweights=True\n'.format(sun_aw, starttime, endtime))

os.system('echo applying beam')
if not dummy:
    os.system('DPPP sun_applybeam.parset')
