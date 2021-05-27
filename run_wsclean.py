#!/usr/bin/env python

"""
Run wsclean for a particular MS and certain intervals
Inputs:
MS
Intervals: pickled pandas dataframe
"""
import argparse
import os

import pandas as pd

from time_to_wsclean_interval import get_interval

def wsclean_command(msin, int0, output):
    if output is None:
        output = 'wsclean'
    wsclean_string = "wsclean -mem 85 -no-reorder -no-update-model-required -mgain 0.6 -weight briggs 0 \
-size 1024 1024 -scale 5asec -pol I -data-column CORRECTED_DATA -taper-gaussian 90 \
-multiscale -niter 320 -fit-beam -interval {} {} -intervals-out 1 -name {} {}".format(int0,
                                                                                      int0+1,
                                                                                      output,
                                                                                      msin)
    return wsclean_string

if __name__ == "__main__"
    parser = argparse.ArgumentParser(description='Convert a time range to data interval for wsclean')
    parser.add_argument('msin', help='input measurement set', metavar='MSIN')
    parser.add_argument('--trange', dest='trange', nargs='+',
                        help='time range for observation.\
                        2 arguments START and END in format YYYY-MM-DDTHH:MM:SS\
                        if only START given then assume END is 1 second later.',
                        metavar=('START', 'END'))
    parser.add_argument('-p', '--pickle', help='pickled pandas dataframe with times of burst', default=None)
    parser.add_argument('-o', '--output', help='name of wsclean output')
    args = parser.parse_args()
    msin = args.msin
    trange = args.trange
    pickle = args.pickle
    output = args.output

    if pickle is not None:
        if len(trange) == 2:
            trange = TimeRange(trange[0], trange[1])  # ("2019-04-04T14:08:00", "2019-04-04T14:17:00")
        elif len(trange) == 1:
            tstart = Time(trange[0])
            trange = TimeRange(tstart, tstart + 1 * u.s)
        else:
            print("Invalid number of trange arguments, 1 or 2")

        int0 = get_interval(msin, trange)[0]
        wsclean_string = wsclean_command(msin, int0, output)
        os.system(wsclean_string)
    else:
        df = pd.read_pickle(pickle)
        for t in df[df.columns[0]]:
            tstart = Time(t)
            trange = TimeRange(tstart, tstart + 1 * u.s)
            int0 = get_interval(msin, trange)[0]
            wsclean_string = wsclean_command(msin, int0, output)
            os.system(wsclean_string)
