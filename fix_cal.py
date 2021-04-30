#! /usr/bin/env python

import os
import sys

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time
from matplotlib import dates
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks, peak_widths

import lofar.parmdb as pb

def find_burst_index(parm, bursttime):
    '''
    find the index in calibration solution array of given burst peak time 
    '''
    bursttime_sec = bursttime.mjd*24*3600
    times = parm['times']
    burst_loc =  np.where(np.abs(times - bursttime_sec) == np.min(np.abs(times - bursttime_sec)))[0][0]
    return burst_loc

def remove_burst(data, burst_loc):
    '''
    go to index in data determined by burst time and remove 8 points either side.
    '''
    data_noburst = np.zeros(len(data)-16)
    data_noburst[:burst_loc-8] = data[:burst_loc-8]
    data_noburst[burst_loc-8:] = data[burst_loc+8:]    
    return data_noburst

def remove_bursts(data, burst_locs):
    '''
    the same as remove_burst() but for a list of indices
    8 works well for calibration solutions from parset
    gaincal.solint=4
    gaincal.timeslotsperparmupdate=500
    '''
    mask = np.zeros_like(data)
    for loc in burst_locs:
        # if loc < 8:
        #     mask[:loc+8] = 1
        # elif loc+8 > len(data):
        #     mask[loc-8:] = 1
        # else:
        mask[loc-8:loc+8] = 1
       
    mask_data = np.ma.array(data, mask=mask)
    data_noburst = mask_data.compressed()

    return data_noburst

def smoothing(parm, bursttime):
    '''
    replace burst in calibration solution with 2nd order polynomial
    '''
    burst_loc = find_burst_index(parm, bursttime)
    times = parm['times']
    #assume quiet 8 time samples either side of the burst
    calsol = parm['values'].reshape(-1)
    calsol_noburst = remove_burst(calsol, burst_loc)
    times_noburst = remove_burst(times, burst_loc)
    interp_func = interp1d(times_noburst, calsol_noburst, 'quadratic')
    calsol_interp = interp_func(times)   

    return calsol_interp

def auto_smooth(parm, std_factor=3, interp_scheme='linear'):
    #same as smoothing but for no input time
    #std_factor is how many standard deviations above the mean to calculate peaks
    #interp_scheme is the 'kind' input for scipy.interpolate.interp1d
    calsol = parm['values'].reshape(-1)
    smooth = np.abs(np.gradient(calsol))
    peaks, properties = find_peaks(smooth, height = np.nanmean(smooth) + std_factor*np.nanstd(smooth))
    # pwidths = properties['']
    times = parm['times']
    calsol_noburst = remove_bursts(calsol, peaks)
    times_noburst = remove_bursts(times, peaks)
    interp_func = interp1d(times_noburst, calsol_noburst, interp_scheme, fill_value='extrapolate')
    # interp_func1 = interp1d(times_noburst, calsol_noburst, 'cubic')
    calsol_interp = interp_func(times)    
    # calsol_interp1 = interp_func1(times) 
    return calsol_interp

def plot_sols(parm, title='', interp_scheme='linear'):
    '''
    plots calibration solution and corrected solution
    inputs: parm = lofar.parmdb object
            title = string, plot_title
            interp_scheme = used in scipy.interpolate.interp1d
    '''
    date_format = dates.DateFormatter("%H:%M:%S")
    calsol = parm['values']
    #if str(title)[14:16] == 'CS':
    new_calsol = auto_smooth(parm, interp_scheme=interp_scheme)
    #else:
    #    new_calsol = auto_smooth(parm, 5, 'linear')
    timearr = Time(parm['times']/24/3600, format='mjd')
    timeplot = timearr.plot_date
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.plot(timeplot, calsol, '-', timeplot, new_calsol, '-')
    ax.set_title(title)
    ax.set_ylabel('magnitude (arbitrary)')
    ax.set_xlabel('Time on {} (UTC)'.format(timearr[0].isot[:10]))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(date_format)
if __name__ == "__main__":
    parser = ArgumentParser("Smooth over bursts in cal solutions")
    parser.add_argument('parmdb', help='Input parmdb. E.g. instrument directory in MS')
    parser.add_argument('-t', '--bursttime', help='Time of burst. Must be of format YYYY-MM-DDTHH:MM:SS.XXX' )
    parser.add_argument('-d', '--dummy', help='Don\'t actually do anything', action='store_true')
    args = parser.parse_args()
    parm = args.parmdb
    bursttime = args.bursttime
    dummy = args.dummy
    if bursttime is not None:
        bursttime = Time(bursttime, format='isot')
    print('---------------------')
    print('loading table')

    print('---------------------')
    
    newparms={}
    parms=pb.parmdb(parm).getValuesGrid("*")
    
    print('---------------------')
    print('resolving corrupted calibration data')
    print('---------------------')
    if not dummy:
        newpb=pb.parmdb(args.parmdb+"_NEW",create=True)
    for key in sorted(parms.keys()):
        newparm=parms[key].copy()
        

        if str(key)[14:16]=='RS':
            newparm['values'] = auto_smooth(newparm, 5)[:,np.newaxis]#newparm['values'] 
        else:
            newparm['values'] = auto_smooth(newparm)[:,np.newaxis]
            #newparm['values'] = smoothing(newparm, bursttime)[:, np.newaxis]


        newparms[key]=newparm


    if not dummy:
        newpb.addValues(newparms)
