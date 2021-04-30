#! /usr/bin/env python

import os
import sys

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time
from matplotlib import dates

import lofar.parmdb as pb

from fix_cal import auto_smooth

parm = 'L700295_SAP001_SB089_uv_aw.MS/instrument'
parm_aof = 'L700295_SAP001_SB089_uv_aw_avaoflag.MS/instrument'

parms=pb.parmdb(parm).getValuesGrid("*:0:0:*")
parms_aof=pb.parmdb(parm_aof).getValuesGrid("*:0:0:*")

# cs001_r = parms['Gain:0:0:Real:CS001LBA']
# cs001_i = parms['Gain:0:0:Imag:CS001LBA']
date_format = dates.DateFormatter("%H:%M:%S")
station0 = 'Gain:0:0:Real:CS001LBA'

real0 = parms[station0]
imag0 = parms[station0.replace('Real', 'Imag')]

real_interp0 = auto_smooth(real0)
imag_interp0 = auto_smooth(imag0)

real_aof0 = parms_aof[station0]
imag_aof0 = parms_aof[station0.replace('Real', 'Imag')]

calsols0 = real0['values'] + 1j*imag0['values']
calsols_interp0 = real_interp0 + 1j*imag_interp0
calsols_aof0 = real_aof0['values'] + 1j*imag_aof0['values']

zoom_start = Time('2019-04-04T12:57:00', format='isot')
zoom_end = Time('2019-04-04T13:00:00', format='isot')


for station in sorted(parms.keys()):
    if 'Imag' in station:
        continue
    real = parms[station]
    imag = parms[station.replace('Real', 'Imag')]

    real_interp = auto_smooth(real)
    imag_interp = auto_smooth(imag)
    
    real_aof = parms_aof[station]
    imag_aof = parms_aof[station.replace('Real', 'Imag')]

    calsols = real['values'] + 1j*imag['values']
    calsols_interp = real_interp + 1j*imag_interp
    calsols_aof = real_aof['values'] + 1j*imag_aof['values']

    timearr = Time(real['times']/24/3600, format='mjd')
    tstart = np.where((timearr > zoom_start) & (timearr < zoom_end))[0][0]
    tend = np.where((timearr > zoom_start) & (timearr < zoom_end))[0][-1]  
    timeplot = timearr.plot_date
    fig, ax = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    ax[0].plot(timeplot[tstart:tend], np.abs(calsols[tstart:tend]), '-', label='no processing')
    # ax[0].plot(timeplot[tstart:tend], np.abs(calsols_interp[tstart:tend]), '-', label='interpolate peaks')
    ax[0].plot(timeplot[tstart:tend], np.abs(calsols_aof[tstart:tend]), '-', label='run aoflagger')
    ax[0].legend()

    ax[1].plot(timeplot[tstart:tend], np.angle(calsols[tstart:tend]), '-', label='no processing')
    # ax[1].plot(timeplot[tstart:tend], np.angle(calsols_interp[tstart:tend])/np.angle(calsols_interp0[tstart:tend]), '-', label='interpolate peaks')
    ax[1].plot(timeplot[tstart:tend], np.angle(calsols_aof[tstart:tend]), '-', label='run aoflagger')
    ax[1].legend()

    fig.suptitle(station[-8:])
    ax[0].set_ylabel('Amplitude (arbitrary)')
    ax[1].set_ylabel('Phase (radians)'.format(station0[-8:]))
    ax[1].set_xlabel('Time on {} (UTC)'.format(timearr[0].isot[:10]))
    ax[1].xaxis_date()
    ax[1].xaxis.set_major_formatter(date_format)
    
    plt.savefig('calsol_comparisons/zoom/calsol_compare_{}_{}_zoom'.format(timearr[0].isot[:10],station[-8:]))
    plt.close()