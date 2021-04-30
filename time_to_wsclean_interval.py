#!/usr/bin/env python

import argparse
import warnings
warnings.filterwarnings("ignore")


import astropy.units as u
import numpy as np

from astropy.time import Time, TimeDelta
from casacore import tables
from sunpy.time import TimeRange

def nbaselines(nants):
    #calculate number of baselines including self correlations
    #given a number of antennas
    return nants*(nants+1)/2


parser = argparse.ArgumentParser(description='Convert a time range to data interval for wsclean')
#should probably get obs_start from the file but that probably needs casacore and stuff
parser.add_argument('msin', help='input measurement set', metavar='MSIN')
parser.add_argument('--trange', dest='trange', nargs='+',
                    help='time range for observation.\
                    2 arguments START and END in format YYYY-MM-DDTHH:MM:SS\
                    if only START given then assume END is 1 second later.',
                    metavar=('START','END'))
args = parser.parse_args()

msin = args.msin
trange = args.trange

if len(trange) == 2:
    trange = TimeRange(trange[0], trange[1])#("2019-04-04T14:08:00", "2019-04-04T14:17:00")
elif len(trange) == 1:
    tstart = Time(trange[0])
    trange = TimeRange(tstart, tstart+1*u.s)
else:
    print("Invalid number of trange arguments, 1 or 2")


# mjd0 = Time(0, format='mjd')
with tables.table(msin, ack=False) as t:
	dt = TimeDelta(t.col('INTERVAL')[0], format='sec')
	t0 = t.getcol('TIME')[0] #seconds from 1858/11/17 MJD reference
	t0 = Time(t0/24/3600, format='mjd') #convert to days

t1 = trange.start-t0
int0 = t1/dt

# int0 = trange.start-obs_start
# int0 = int(np.floor(int0/dt))
int0 = int(np.floor(int0))

int_len = trange.dt/dt
int_len = int(np.floor(int_len))
int1 = int0 + int_len

print("interval is: {} to {} (ish)".format(int0, int1))
