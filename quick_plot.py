#!/usr/bin/env python

import argparse

import astropy.units as u
import matplotlib.pyplot as plt

from sunpy.time import TimeRange

from LOFAR_bf import LOFAR_BF

parser = argparse.ArgumentParser(description='quickly plot from input h5 file')

parser.add_argument('bffile', help='input name of beamformed data')
parser.add_argument('-s','--save',action='store_true',
                    help='set to save figure, default = False')
parser.add_argument('-t', '--trange', help='Time range to plot. \
                    2 arguments, must be of format YYYY-MM-DDTHH:MM:SS',
                    nargs=2, metavar=('START','END'))
parser.add_argument('-f', '--frange', help='Frequency range to plot. \
                    2 arguments, must be given in MHz.',
                    nargs=2, type=int, metavar=('F0','F1'))
parser.add_argument('--show', action='store_true', help='If True, shows plot \
                    default=False')
args = parser.parse_args()

bffile = args.bffile
save = args.save
trange = args.trange
frange = args.frange
show = args.show
if trange is not None:
    trange = TimeRange(trange[0], trange[1])
if frange is not None:
    frange = frange*u.MHz

bf = LOFAR_BF(bffile, trange, frange)
clip =  [5,95]*u.percent
print('Plotting ' + bffile)
fig, ax = plt.subplots(figsize=(10,7))
bf.plot(ax, bg_subtract=True, clip_interval=clip)
if save:
    save_path = trange.start.isot[:-4].replace('-', '').replace(':','') + \
                '_' + trange.end.isot[11:-4].replace(':','')
    print('Saving to ' + save_path)
    plt.savefig(save_path+'.png')
if show:
    plt.show()
else:
    plt.close()
