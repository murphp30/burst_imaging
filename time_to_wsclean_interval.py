#!/usr/bin/env python

import argparse
import warnings


import astropy.units as u
import numpy as np
import pandas as pd

from astropy.time import Time, TimeDelta
from casacore import tables
from sunpy.time import TimeRange

warnings.filterwarnings("ignore")


def nbaselines(nants):
    # calculate number of baselines including self correlations
    # given a number of antennas
    return nants * (nants + 1) / 2


# mjd0 = Time(0, format='mjd')
def get_interval(msin, trange):
    """
    Convert sunpy.time.Timerange to interval for WSClean
    Inputs:
    msin: input Measurement Set (MS)
    trange: time as sunpy.time.Timerange
    """
    with tables.table(msin, ack=False) as t:
        dt = TimeDelta(t.col('INTERVAL')[0], format='sec')
        t0 = t.getcol('TIME')[0]  # seconds from 1858/11/17 MJD reference
        t0 = Time(t0 / 24 / 3600, format='mjd')  # convert to days
    if trange.start < t0:
        print("trange start is before observation start, interval will start at 0")
        int0 = 0
    else:
        t1 = trange.start - t0
        int0 = t1 / dt

        # int0 = trange.start-obs_start
        # int0 = int(np.floor(int0/dt))
        int0 = int(np.round(int0))

    int_len = trange.dt / dt
    int_len = int(np.round(int_len))
    int1 = int0 + int_len

    print("interval: {} {}".format(int0, int1))
    return int0, int1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a time range to data interval for wsclean')
    # should probably get obs_start from the file but that probably needs casacore and stuff
    parser.add_argument('msin', help='input measurement set', metavar='MSIN')
    parser.add_argument('--trange', dest='trange', nargs='+',
                        help='time range for observation.\
                        2 arguments START and END in format YYYY-MM-DDTHH:MM:SS\
                        if only START given then assume END is 1 second later.',
                        metavar=('START', 'END'))
    parser.add_argument('-p', '--pickle', help='pickled pandas dataframe with times of burst', default=None)
    args = parser.parse_args()

    msin = args.msin
    trange = args.trange
    pickle = args.pickle

    if pickle is None:
        if len(trange) == 2:
            trange = TimeRange(trange[0], trange[1])  # ("2019-04-04T14:08:00", "2019-04-04T14:17:00")
        elif len(trange) == 1:
            tstart = Time(trange[0])
            trange = TimeRange(tstart, tstart + 1 * u.s)
        else:
            print("Invalid number of trange arguments, 1 or 2")

        get_interval(msin, trange)
    else:
        df = pd.read_pickle(pickle)
        for t in df[df.columns[0]]:
            tstart = Time(t)
            trange = TimeRange(tstart, tstart + 0.1 * u.s)
            get_interval(msin, trange)

