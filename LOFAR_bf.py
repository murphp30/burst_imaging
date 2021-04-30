#!/usr/bin/env/python

import copy

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time, TimeDelta
from astropy.visualization import AsymmetricPercentileInterval
from matplotlib import dates
from sunpy.time import TimeRange

class LOFAR_BF:
    """
    Class to read LOFAR beamformed data downloaded off the LTA.
    Inputs: 
        bf_file = Filename for h5 file. MUST be in same directory as raw data file.
        trange = Time range (sunpy.time.TimeRange)
        frange = Frequency range (2 element list or array as astropy.units.Quantity)
    """
    def __init__(self, bf_file, trange=None, frange=None):
        self.bf_file = bf_file
        self.trange = trange
        self.frange = frange
        self.__get_data()
    def __get_data(self):
        fname = self.bf_file.split('/')[-1].split('_')
        SAP = fname[1] #Sub-Array pointing
        beam = fname[2] #Beam number
        stokes = fname[3] #Stokes parameter
        with h5py.File(self.bf_file, 'r') as h5:
            self.obs_start = Time(h5['SUB_ARRAY_POINTING_'+SAP[-3:]].attrs['EXPTIME_START_UTC'])
            self.obs_end = Time(h5['SUB_ARRAY_POINTING_'+SAP[-3:]].attrs['EXPTIME_END_UTC'])
            
            sap_ra = Angle(h5['SUB_ARRAY_POINTING_'+SAP[-3:]].attrs['POINT_RA'],
                           h5['SUB_ARRAY_POINTING_'+SAP[-3:]].attrs['POINT_RA_UNIT'])
            sap_dec = Angle(h5['SUB_ARRAY_POINTING_'+SAP[-3:]].attrs['POINT_DEC'],
                            h5['SUB_ARRAY_POINTING_'+SAP[-3:]].attrs['POINT_DEC_UNIT'])
            self.sap_skycoord = SkyCoord(sap_ra, sap_dec, frame='icrs', obstime=self.obs_start, equinox='J2000')

            beam_ra = Angle(h5['SUB_ARRAY_POINTING_' + SAP[-3:] + '/BEAM_' + beam[-3:]].attrs['POINT_RA'],
                            h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:]].attrs['POINT_RA_UNIT'])
            beam_dec = Angle(h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:]].attrs['POINT_DEC'],
                             h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:]].attrs['POINT_DEC_UNIT'])
            self.beam_skycoord = SkyCoord(beam_ra, beam_dec, frame='icrs', obstime=self.obs_start, equinox='J2000')
            
            dt_unit = h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:]].attrs['SAMPLING_TIME_UNIT']
            if dt_unit == 's':
                fmt = 'sec'
            self.dt = TimeDelta(h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:]].attrs['SAMPLING_TIME'], format=fmt)
            self.n_samples = h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:]].attrs['NOF_SAMPLES']
            self.n_channels = h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:]].attrs['CHANNELS_PER_SUBBAND'] *\
                         h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:] + '/STOKES_' + stokes[1]].attrs['NOF_SUBBANDS']
            
            self.freq_unit = u.Quantity(1,h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:] + '/COORDINATES/COORDINATE_1'].attrs['AXIS_UNITS'][0]) 
            freq_arr = h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:] + '/COORDINATES/COORDINATE_1'].attrs['AXIS_VALUES_WORLD'] * self.freq_unit
            if self.frange is None and self.trange is None:
                self.freqs = freq_arr
                self.data = h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:] + '/STOKES_' + stokes[1]][:,:]
            
            else:
                if self.frange is not None and self.trange is not None:
                    f0 = np.where(abs(self.frange[0] - freq_arr) == np.min(abs(self.frange[0] - freq_arr)))[0][0]
                    f1 = np.where(abs(self.frange[1] - freq_arr) == np.min(abs(self.frange[1] - freq_arr)))[0][0]
                    self.freqs = freq_arr[f0:f1]

                    t0 = int(np.floor((self.trange.start - self.obs_start).sec/self.dt.sec))
                    t1 = int(np.floor((self.trange.end - self.obs_start).sec/self.dt.sec))
                    self.data = h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:] + '/STOKES_' + stokes[1]][t0:t1,f0:f1]
                
                elif self.frange is None:
                    t0 = int(np.floor((self.trange.start - self.obs_start).sec/self.dt.sec))
                    t1 = int(np.floor((self.trange.end - self.obs_start).sec/self.dt.sec))
                    self.freqs = freq_arr
                    self.data = h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:] + '/STOKES_' + stokes[1]][t0:t1,:]

                elif self.trange is None:
                    f0 = np.where(abs(self.frange[0] - freq_arr) == np.min(abs(self.frange[0] - freq_arr)))[0][0]
                    f1 = np.where(abs(self.frange[1] - freq_arr) == np.min(abs(self.frange[1] - freq_arr)))[0][0]
                    self.freqs = freq_arr[f0:f1]
                    self.data = h5['SUB_ARRAY_POINTING_' + SAP[-3:]+'/BEAM_' + beam[-3:] + '/STOKES_' + stokes[1]][:,f0:f1]

    def bg(self, amount=0.05):
        #adapted from radiospectra.Spectrogram.auto_find_background()
        #doesn't do mean subtraction at the beginning
        tmp = copy.deepcopy(self.data)# - np.mean(data, 1)[:, np.newaxis]
        sdevs = np.std(tmp, 1)
        cand = sorted(range(tmp.shape[0]), key=lambda y: sdevs[y])
        realcand = cand[:max(1, int(amount*len(cand)))]
        bg = np.mean(tmp[realcand,:], 0)
        return bg[np.newaxis, :]
    
    def plot(self, ax=None, title=None, bg_subtract=False, scale="linear", clip_interval: u.percent=None):
        """
        Plot dynamic spectrum of data
        Arguments:
            ax = matplotlib axis
            title = str, plot title
            bg_subtract = bool. If Ture, perform background subtraction from data.
            scale = str, "linear" or "log" for colour scale
            clip_interval = astropy.units.Quantity sets vmin and vmax for plot. e.g. Quantity([1,99]*u.percent)
        """
        imshow_args = {}
        data = copy.deepcopy(self.data)
        if not ax:
            fig, ax = plt.subplots()
        
        if bg_subtract:
            data = (data/self.bg())

        if scale == 'log':
            data = np.log10(data)

        if clip_interval is not None:
            if len(clip_interval) == 2:
                clip_percentages = clip_interval.to('%').value
                vmin, vmax = AsymmetricPercentileInterval(*clip_percentages).get_limits(data)
            else:
                raise ValueError("Clip percentile interval must be specified as two numbers.")
            imshow_args["vmin"] = vmin
            imshow_args["vmax"] = vmax

        if self.trange is None:
            ret = ax.imshow(data.T, aspect="auto", 
                            extent = [self.obs_start.plot_date, self.obs_end.plot_date, self.freqs.to(u.MHz).value[-1], self.freqs.to(u.MHz).value[0]],
                            **imshow_args)
        else:
            ret = ax.imshow(data.T, aspect="auto",
                            extent = [self.trange.start.plot_date, self.trange.end.plot_date, self.freqs.to(u.MHz).value[-1], self.freqs.to(u.MHz).value[0]],
                            **imshow_args)
        ax.xaxis_date()
        date_format = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(date_format)

        if title:
            ax.set_title(title)
        ax.set_xlabel("Start Time " + self.obs_start.iso[:-4] + "UTC")
        ax.set_ylabel("Frequency (MHz)")

        return ret
