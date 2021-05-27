#!/usr/bin/env python

import argparse

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import savgol_filter, find_peaks
from sunpy.time import TimeRange

from LOFAR_bf import LOFAR_BF


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rebin_array(arr, new_index):
    return np.array(
        [np.sum(arr[i * new_index:(i + 1) * new_index]) for i in range(int(np.round(len(arr) / new_index)))])


parser = argparse.ArgumentParser(description='Find a number of peaks in a LOFAR LTA dynamic spectrum')
parser.add_argument('f', help='filename of h5 LTA file.', metavar='FILENAME')
parser.add_argument('-t', '--trange', nargs=2,
                    help='time range for observation.2 arguments START and END in format YYYY-MM-DDTHH:MM:SS',
                    metavar=('START', 'END'))
parser.add_argument('-p', '--plot',
                    help='plot dynamic spectrum. Default is True',
                    action='store_true')
args = parser.parse_args()
f = args.f
trange = args.trange
plot = args.plot

if trange is not None:
    trange = TimeRange(trange[0], trange[1])  # ("2019-04-04T14:08:00", "2019-04-04T14:17:00")
clip = u.Quantity([5, 95] * u.percent)

bf = LOFAR_BF(f, trange)
tarr = np.arange(bf.data.shape[0]) * bf.dt
tarr = tarr + bf.trange.start
new_dt = 0.167772  # 0.167772 seconds is interferometric temporal resolution
new_dt_index = int(np.round(new_dt / bf.dt.sec))
sig = new_dt_index / (2 * np.sqrt(2 * np.log(2)))  # FWHM is interferometric time res.
freqs = [51]  # np.arange(20,80,10)
if plot:
    fig, ax = plt.subplots(figsize=(10, 8))
    bf.plot(ax=ax, bg_subtract=True, clip_interval=clip)
peaks_df = []#pd.DataFrame()
max_peaks_df = []
for freq in freqs:
    loc = np.where(abs(bf.freqs.to(u.MHz) - freq * u.MHz) == np.min(abs(bf.freqs.to(u.MHz) - freq * u.MHz)))[0][0]
    dslice = np.mean(bf.data[:, loc - 8:loc + 8], axis=1)  # bf.data[:, loc] average over 16 channels
    dslice = rebin_array(dslice, new_dt_index)
    tarr = tarr[new_dt_index//2::new_dt_index]
    smooth = gaussian_filter1d(dslice, sig)
    # define a background using rolling window of standard deviations
    win_len = int(np.round(10 / bf.dt.sec))  # 10 second window
    stds = np.std(rolling_window(smooth, win_len), 1)
    bg_std = stds[np.where(stds == np.min(stds))[0][0]]
    means = np.mean(rolling_window(smooth, win_len), 1)
    bg_mean = means[np.where(stds == np.min(stds))[0][0]]

    peaks, _ = find_peaks(smooth,
                          height= np.mean(dslice) + 3 * bg_std,
                          distance= 4 * new_dt_index)#,
                          # prominence= bg_std)
    # width = new_dt_index)
    # prominence=(np.max(smooth)-np.mean(smooth))/10)
    # peaks, _ = find_peaks(dslice,
    #        height=np.mean(dslice)+3*np.std(dslice))
    # prominence=(np.max(smooth)-np.mean(smooth))/10)

    # ax.scatter(tarr.plot_date[peaks],
    #         np.ones(len(peaks))*bf.freqs[loc].to(u.MHz).value,
    #         color='r', marker='+')
    # max_peak = peaks[np.where(dslice[peaks] == np.max(dslice[peaks]))[0][0]]
    max_peak = peaks[np.where(smooth[peaks] == np.max(smooth[peaks]))[0][0]]
    max_peak_time = tarr[max_peak]
    peak_df = pd.DataFrame(tarr[peaks], columns=[bf.freqs[loc].to(u.MHz).value])
    peaks_df.append(peak_df) #= peaks_df.append(peak_df)
    print('Maximum peak at {}'.format(max_peak_time))
    max_peak_df = pd.DataFrame([max_peak_time.isot], columns=[bf.freqs[loc].to(u.MHz).value])
    max_peaks_df.append(max_peak_df)  # = peaks_df.append(peak_df)
    if plot:
        ax.scatter(tarr.plot_date[peaks],
                   np.ones(len(peaks)) * bf.freqs[loc].to(u.MHz).value,
                   color='w', marker='o')
        ax.scatter(tarr.plot_date[max_peak],
                   bf.freqs[loc].to(u.MHz).value,
                   color='r', marker='+')

# fig, ax = plt.subplots(figsize=(13, 10))
# ax.plot(tarr.plot_date, dslice)
# # plt.plot(tarr.plot_date, smooth)
# ax.vlines(tarr.plot_date[max_peak], np.min(dslice), dslice[max_peak], color='r', zorder=1000)
# ax.xaxis_date()
# date_format = dates.DateFormatter("%H:%M:%S")
# ax.xaxis.set_major_formatter(date_format)
# plt.hlines(np.mean(dslice) + 5 * bg_std, 0, len(smooth), color='k', zorder=1000)
save_path = "./"  # "/mnt/murphp30_data/paper2"
save_png = save_path + "/peak_times_{}_{}.png".format(trange.start.isot[:-4].replace(':', ''),
                                                      trange.end.isot[11:-4].replace(':', ''))
# if plot:
#     plt.savefig(save_png)
peaks_df = pd.concat(peaks_df, axis=1)
max_peaks_df = pd.concat(max_peaks_df, axis=1)
save_pickle = save_path + "/peak_times_{}_{}.pickle".format(trange.start.isot[:-4].replace(':', ''),
                                                            trange.end.isot[11:-4].replace(':', ''))
# with open("peak_times.txt", 'a') as peak_file:
#     peak_file.write(max_peak_time.isot)
#     peak_file.write('\n')
peaks_df.to_pickle(save_pickle)
# max_peaks_df.to_pickle(save_pickle)
# plt.close()
plt.show()
