#!/usr/bin/env python

"""
Use visibility fitting from plot_vis.py to find the area of a fitted source
"""

import argparse
import asyncio
import sys

from multiprocessing import Pool
from itertools import product

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.coordinates import Angle
from astropy.time import Time
from lmfit.models import LinearModel
from sunpy.time import TimeRange

sys.path.insert(1, '/mnt/LOFAR-PSP/pearse_2ndperihelion/scripts')
sys.path.insert(1, '/Users/murphp30/mnt/LOFAR-PSP/pearse_2ndperihelion/scripts')

from plot_vis import fit_burst, LOFAR_vis

def calculate_areas(fit_df, time_from_start):

    sig_xs = np.array([fit.params['sig_x'].value for fit in fit_df[0]])
    sig_ys = np.array([fit.params['sig_y'].value for fit in fit_df[0]])
    areas = Angle(sig_xs * u.rad) * Angle(sig_ys * u.rad) * np.pi

    sig_x_std = np.array([fit.params['sig_x'].stderr for fit in fit_df[0]])
    deltas_std = np.array([fit.params['delta'].stderr for fit in fit_df[0]])
    sig_y_std = np.sqrt(sig_x_std**2 + deltas_std**2)
    area_std = areas * np.sqrt((sig_x_std / sig_xs) ** 2 + (sig_y_std / sig_ys) ** 2)

    areas_amin = areas.to(u.arcmin ** 2)
    area_std_amin = area_std.to(u.arcmin ** 2)

    area_growth = LinearModel()
    pars = area_growth.guess(areas_amin, x=time_from_start)
    area_growth_fit = area_growth.fit(areas_amin.value, pars, x=time_from_start, weights=1/area_std_amin.value)

    return areas_amin, area_std_amin, area_growth_fit

def plot_areas(fit_df, time_from_start):
    I0s = [fit.params['I0'].value for fit in fit_df[0]]

    plt.figure()
    plt.plot(time_from_start, I0s, 'o')
    plt.xlabel("Time (s)")
    plt.ylabel("Peak Intensity")

    areas_amin, area_std_amin, area_growth_fit = calculate_areas(fit_df, time_from_start)
    plt.figure()
    plt.errorbar(time_from_start, areas_amin.value,
                 area_std_amin.value, ls='', marker='o')
    plt.plot(time_from_start, area_growth_fit.best_fit)
    plt.xlabel("Time (s)")
    plt.ylabel(r"Area (arcmin$^2$)")

"""
async def multi_fit_burst(vis):
    # with Pool() as pool:
    #     fit_burst_time_dir = np.array(pool.starmap(fit_burst, product([vis], range(len(vis.time)))))
    # fit_burst_time_dir = np.array([])
    task_list = []
    for i in range(len(vis.time)):
        task_list.append(asyncio.create_task(fit_burst(vis,i)))

    await asyncio.gather(fit_burst(vis,0), fit_burst(vis,1))
    # await task_list[1]


    # fit_burst_time_dir = await asyncio.gather(*task_list)

    fit_burst_time_dir = fit_burst_time_dir.T
    fit, burst_time, burst_centre_coord = fit_burst_time_dir
    fit_df = pd.DataFrame(fit, index=burst_time)
    fit_df['burst_centre_coord'] = burst_centre_coord
    return fit_df

async def main(msin, trange_list):
    # task_list = []
    # for trange in trange_list:
    #     vis = LOFAR_vis(msin, trange)
    #     task = asyncio.create_task(multi_fit_burst(vis))
    #     # await task
    #     task_list.append(task)
    # await asyncio.gather(*task_list)
    vis = LOFAR_vis(msin, trange_list[0])
    task = asyncio.create_task(multi_fit_burst(vis))
    await task
"""

def main(msin, trange):
    vis = LOFAR_vis(msin, trange)
    with Pool() as pool:
        fit_burst_time_dir = np.array(pool.starmap(fit_burst, product([vis], range(len(vis.time)))))
    fit_burst_time_dir = fit_burst_time_dir.T
    fit, burst_time, burst_centre_coord = fit_burst_time_dir
    fit_df = pd.DataFrame(fit, index=burst_time)
    fit_df['burst_centre_coord'] = burst_centre_coord
    time_from_start = (vis.time - vis.time[0]).sec
    areas_amin, area_std_amin, area_growth_fit = calculate_areas(fit_df, time_from_start)
    # area_dict = {"areas_amin":areas_amin, "areas_std_amin":area_std_amin, "area_growth_fit":area_growth_fit}
    # area_df = pd.DataFrame.from_dict(area_dict, index=)
    return areas_amin, area_std_amin, area_growth_fit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('MS', help='Input measurement set.')
    parser.add_argument('--trange', dest='trange', nargs='+',
                        help='time range for observation.\
                        2 arguments START and END in format YYYY-MM-DDTHH:MM:SS\
                        if only START given then assume END is 1 second later.',
                        metavar=('START', 'END'))
    parser.add_argument('-p', '--pickle', default=None,
                        help='Name of pickle file with list of times')
    args = parser.parse_args()
    msin = args.MS
    pickle = args.pickle
    if pickle is None:
        trange = args.trange
        if len(trange) == 2:
            trange = TimeRange(trange[0], trange[1])
        elif len(trange) == 1:
            #make bold assumption of onset and offset times of burst
            tstart = Time(trange[0])
            trange = TimeRange(tstart - 1 * u.s, tstart + 2 * u.s)

        vis = LOFAR_vis(msin, trange)
        with Pool() as pool:
            fit_burst_time_dir = np.array(pool.starmap(fit_burst, product([vis], range(len(vis.time)))))
        fit_burst_time_dir = fit_burst_time_dir.T
        fit, burst_time, burst_centre_coord = fit_burst_time_dir
        fit_df = pd.DataFrame(fit, index=burst_time)
        fit_df['burst_centre_coord'] = burst_centre_coord

        time_from_start = (vis.time - vis.time[0]).sec
        plot_areas(fit_df, time_from_start)
        plt.show()
        #
    else:
        df = pd.read_pickle(pickle)
        tstart_list = []
        area_dict = {"areas_amin": [], "area_std_amin": [], "area_growth_fit": []}
        for i, t in enumerate(df[df.columns[0]]):
            tstart = Time(t)
            dummy_vis = LOFAR_vis(msin, TimeRange(tstart, tstart+0.16*u.s)) #weird off-by-one error between times and time in MS
            tstart_list.append(dummy_vis.time.isot[0])
            #same bold assumption of burst time as above
            trange = TimeRange(tstart - 1 * u.s, tstart + 2 * u.s)

            areas_amin, area_std_amin, area_growth_fit = main(msin, trange)
            area_dict["areas_amin"].append(areas_amin)
            area_dict["area_std_amin"].append(area_std_amin)
            area_dict["area_growth_fit"].append(area_growth_fit)
        area_df = pd.DataFrame(area_dict, index=tstart_list)
        area_df.to_pickle("area_df_{}.pkl".format(trange.start.isot[:10]))