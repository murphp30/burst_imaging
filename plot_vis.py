#!/usr/bin/env python
"""
Visibility fitting script for LOFAR interferometric data.
Fits a single elliptical gaussian in visibility space
Inputs: Measurement Set
        Time range or pickled pandas dataframe of time ranges

"""

import argparse
import os
import pdb
import sys
import time

from multiprocessing import Pool

import corner
import emcee

import astropy.units as u
import matplotlib.colorbar as colorbar
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy.map

from itertools import product

from astropy.constants import au, e, eps0, c, m_e, R_sun
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.time import Time, TimeDelta
from casacore import tables
from lmfit import Model, Parameters, minimize, report_fit
from lmfit.models import LinearModel
from matplotlib import dates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from sunpy.coordinates import frames, sun
from sunpy.sun.constants import average_angular_size as R_av_ang
from sunpy.time import TimeRange

sys.path.insert(1, '/mnt/LOFAR-PSP/pearse_2ndperihelion/scripts')
sys.path.insert(1, '/Users/murphp30/mnt/LOFAR-PSP/pearse_2ndperihelion/scripts')
plt.style.use('seaborn-colorblind')

from icrs_to_helio import icrs_map_to_helio

def sig_to_fwhm(sig):
    """
    Converts standard deviation to Full Width at Half Maximum height
    """
    fwhm = (2 * np.sqrt(2 * np.log(2)) * sig)
    fwhm = Angle(fwhm * u.rad).arcmin
    return fwhm


def fwhm_to_sig(fwhm):
    """
    Converts Full Width at Half Maximum height to standard deviation
    """
    c = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return c


def rotate_coords(u, v, theta):
    """
    Rotate coordinates (u, v)  anticlockwise by theta
    """

    u_p = u * np.cos(theta) + v * np.sin(theta)
    v_p = -u * np.sin(theta) + v * np.cos(theta)
    return u_p, v_p

def fourier_shift(u, v, x0, y0):
    shift = np.exp(-2 * np.pi * 1j * (u * x0 + v * -y0))
    return shift

def gauss_amp(u, v, I0, sig_x, sig_y, theta):
    u_p, v_p = rotate_coords(u, v, -theta)
    V = I0 * np.exp(
        -((((sig_x ** 2) * ((2 * np.pi * u_p) ** 2)) / 2) + (((sig_y ** 2) * ((2 * np.pi * v_p) ** 2)) / 2)))
    return V

def gauss_2D(u, v, I0, x0, y0, sig_x, sig_y, theta):
    """
    Create a 2D Gaussian in visibility space
    Inputs: u: array of u coordinates
            v: array of V coordinates
            I0: maximum amplitude
            x0: offset of source in x direction from centre in real space
            y0: offset of source in y direction from centre in real space
            sig_x: standard deviation of gaussian along x axis in real space
            sig_y: standard deviation of gaussian along y axis in real space
            theta: position angle of gaussian
            C: offset from 0
    """
    # pa = -theta  # reverse gauss position angle because we're rotating the gaussian to align with our coordinates
    # it doesn't work if you don't so that's reason enough...

    # u_p, v_p = rotate_coords(u, v, pa)
    # x0_p, y0_p = rotate_coords(x0, y0, pa)
    # amp = I0 #/ (2 * np.pi)

    shift = fourier_shift(u, v, x0, y0)#np.exp(-2 * np.pi * 1j * (u * x0 + v * -y0))
    amp = gauss_amp(u, v, I0, sig_x, sig_y, theta)
    V = shift * amp #* np.exp(
        # -((((sig_x ** 2) * ((2 * np.pi * u_p) ** 2)) / 2) + (((sig_y ** 2) * ((2 * np.pi * v_p) ** 2)) / 2)))
    return V

def gauss_2D_real(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
    (x, y) = xy
    x, y = rotate_coords(x, y, theta)
    x0, y0 = rotate_coords(x0, y0 , theta)
    g = amp * np.exp(-(((x - x0) ** 2) / (2 * sig_x ** 2) + ((y - y0) ** 2) / (2 * sig_y ** 2))) + offset
    return g.ravel()

def residual(params, u, v, data, weights=None, fit="amplitude"):
    """
    Residual function for lmfit.minimizer()
    Inputs: params: lmfit.parameter.Parameters inputs for gaussian model
            u: u coordinates of data
            v: v coordinates of data
            data: data to fit
            weights: array the same shape as data of measurement uncertainties
    """
    params_dict = params.valuesdict()
    # pdb.set_trace()
    if fit == "amplitude":
        model = gauss_amp(u.data.value, v.data.value, params_dict['I0'], params_dict['sig_x'], params_dict['sig_y'], params_dict['theta'])
        # model = gauss_2D(u.value, v.value,
        #                  params_dict['I0'], 0, 0,
        #                  params_dict['sig_x'], params_dict['sig_y'], 0, 0)
        if weights is None:
            resid = np.abs(data) - np.abs(model)
        else:
            resid = (np.abs(data) - model) * weights
    elif fit == "phase":
        model = fourier_shift(u.data.value, v.data.value, params_dict['x0'], params_dict['y0'])
        # model = gauss_2D(u.value, v.value,
        #                  params_dict['I0'], params_dict['x0'], params_dict['y0'],
        #                  params_dict['sig_x'], params_dict['sig_y'], 0, 0)
        if weights is None:
            resid = np.angle(data) - np.angle(model)
        else:
            resid = (np.angle(data) - np.angle(model)) * weights
    elif fit == "all":
        model = gauss_2D(u.data.value, v.data.value,
                         params_dict['I0'], params_dict['x0'], params_dict['y0'],
                         params_dict['sig_x'], params_dict['sig_y'], params_dict['theta'])
        if weights is None:
            resid = np.abs(data - model) + np.angle(data - model)#np.sqrt((np.abs(data - model))**2 + (np.angle(data - model))**2)
        else:
            resid = (np.abs(data - model) + np.angle(data - model))* weights
            #np.abs((data - model)) * weights
                # np.sqrt((np.abs(data - model))**2 + (np.angle(data - model))**2)*weights
            #np.sqrt((data.real - model.real)**2 + (data.imag - model.imag)**2) * weights#np.abs((data - model)) * weights
    else:
        print("Invalid residul please choose either 'amplitude', 'phase' or 'all'")
        resid = None
    return resid


def nbaselines(nants):
    """
    Calculate number of baselines including self correlations given a number of antennas
    """
    return nants * (nants + 1) / 2


class LOFAR_vis:
    """
    A class that contains a LOFAR measurment set
    and the various ways it's split into useful things.
    Inputs: fname: file name of measurement set
            trange: sunpy.time.Timerange time range of interest

    Attributes: antenna1 = array of 1st antennas in each baseline
                antenna2 = array of 2nd antennas in each baseline
                data = array of calibrated data
                dt = time resolution of observation
                flag = array of flagged baselines
                fname = file name of Measurement Set (MS)
                freq = frequency of observation
                model = array of model data
                nbaselines = number of baselines in MS
                phase_dir = Phase direction (ra, dec) of LOFAR observation
                time = array of times
                trange = Timerange of data
                uncal = array of uncalibrated data
                uvw = uvw coordinates of observation
                weight = array of data weights
                wlen = wavelength of observation

    Methods: stokes()
    """

    def __init__(self, fname, trange):
        self.fname = fname
        self.trange = trange
        with tables.table(self.fname + 'ANTENNA', ack=False) as tant:
            self.nbaselines = nbaselines(tant.nrows())
        with tables.table(self.fname + 'SPECTRAL_WINDOW', ack=False) as tspec:
            freq = tspec.col('REF_FREQUENCY')[0] * u.Hz
            self.wlen = (c / freq).decompose()
            self.freq = freq.to(u.MHz)
        self.__get_data()
        with tables.table(self.fname + 'FIELD', ack=False) as tfield:
            phase_dir = tfield.col('PHASE_DIR')[0][0]
            core_ITRF = np.array((3826577.462, 461022.624, 5064892.526))
            lofar_loc = EarthLocation.from_geocentric(*core_ITRF, u.m)
            self.lofar_gcrs = SkyCoord(lofar_loc.get_gcrs(Time(self.time.isot)))
            self.phase_dir = SkyCoord(*phase_dir * u.rad,
                                      frame='gcrs',
                                      obstime=self.time.isot,
                                      obsgeoloc=self.lofar_gcrs.cartesian,
                                      obsgeovel=self.lofar_gcrs.velocity.to_cartesian(),
                                      distance=self.lofar_gcrs.hcrs.distance,
                                      equinox='J2000')


    def __get_data(self):
        with tables.table(self.fname, ack=False) as t:
            self.dt = t.col('INTERVAL')[0] * u.s
            #need to correct below for off-by-one error compared to time_to_wsclean_interval.get_interval
            ts = (self.trange.start - 0.16*u.s).mjd * 24 * 3600
            te = (self.trange.end - 0.16*u.s).mjd * 24 * 3600

            with tables.taql('SELECT * FROM $t WHERE TIME > $ts AND TIME < $te') as t1:
                antenna1 = t1.getcol('ANTENNA1')
                antenna2 = t1.getcol('ANTENNA2')
                cross_cors = np.where(antenna1 != antenna2)
                self.time = Time(t1.getcol('TIME', rowincr=int(self.nbaselines)) / 24 / 3600, format='mjd')
                self.flag = t1.getcol('FLAG')[:, 0, :][cross_cors]
                self.antenna1 = np.ma.array(antenna1[cross_cors], mask=self.flag[:,0])
                self.antenna2 = np.ma.array(antenna2[cross_cors], mask=self.flag[:,0])
                uvw = np.ma.array(t1.getcol('UVW')[cross_cors] * u.m / self.wlen, mask=self.flag[:,:3])
                self.uvw = uvw.reshape(len(self.time), -1, 3)
                data = np.ma.array(t1.getcol('CORRECTED_DATA')[:, 0, :][cross_cors], mask=self.flag)
                self.data = data.reshape(len(self.time), -1, 4)
                uncal = np.ma.array(t1.getcol('DATA')[:, 0, :][cross_cors], mask=self.flag)
                self.uncal = uncal.reshape(len(self.time), -1, 4)
                model = np.ma.array(t1.getcol('MODEL_DATA')[:, 0, :][cross_cors], mask=self.flag)
                self.model = model.reshape(len(self.time), -1, 4)
                weight = np.ma.array(t1.getcol('WEIGHT_SPECTRUM')[:, 0, :][cross_cors], mask=self.flag)
                self.weight = weight.reshape(len(self.time), -1, 4)

    def stokes(self, param, t=0):
        """
        Returns inputed Stokes Parameter of either I, Q, U or V (upper case only)
        """
        accepted_params = ['I', 'Q', 'U', 'V']
        if param not in accepted_params:
            print("Please choose one of: I, Q, U, V.")
            return
        if param == 'I':
            return self.data[t, :, 0] + self.data[t, :, 3]
        elif param == 'Q':
            return self.data[t, :, 0] - self.data[t, :, 3]
        elif param == 'U':
            return np.real(self.data[t, :, 1] + self.data[t, :, 2])
        elif param == 'V':
            return np.imag(self.data[t, :, 1] - self.data[t, :, 2])

    def plot(self):
        """
        I don't remember why I wrote this
        Outputs a plot of the uv coverage for the MS
        """
        uv_dist = np.sqrt(self.uvw[:, 0] ** 2 + self.uvw[:, 1] ** 2)
        ang_scales = Angle((1 / uv_dist) * u.rad)
        plot_data = self.stokes('I').data
        ang_scales = ang_scales.reshape(len(self.time), -1)
        plot_data = plot_data.reshape(len(self.time), -1)
        plt.figure()
        plt.plot(ang_scales.arcmin, np.abs(plot_data), 'o')
        plt.title('Amplitude vs Angular Scale')
        plt.ylabel('Amplitude (arbitrary)')
        plt.xlabel('Angular scale (arcmin)')
        plt.xscale('log')
        plt.show()

def briggs(w_i, R):
    """
    Hacky implementation of Briggs robustness weighting
    https://casa.nrao.edu/Release4.1.0/doc/UserMan/UserMansu262.html
    """
    W_k = (np.max(w_i)/w_i)
    f_sq = (5*(10**-R))/(np.sum(W_k**2)/np.sum(w_i))
    return w_i/(1+W_k*f_sq)

def plot_fit(vis, data, fit, plot=True, save=True, outdir="vis_fits/30MHz/", t=0):
    """
    Plots data and fit to data, MCMC corner plot and the chain itself
    Inputs: vis = LOFAR_vis object for burst
            data = data used in fit (probably a better way of getting this because it's
            technically part of vis but I didn't want to change things twice while I was messing around with it)
            fit = lmfit.minimizer.MinimizerResult of MCMC fit
            plot = boolean. False runs plt.close() after each plot, default = True/
    """
    us = np.arange(np.min(vis.uvw[t, :, 0].data.value), np.max(vis.uvw[t, :, 0].data.value), 10)
    vs = np.arange(np.min(vis.uvw[t, :, 1].data.value), np.max(vis.uvw[t, :, 1].data.value), 10)
    uu, vv = np.meshgrid(us, vs)
    fit_data = gauss_2D(uu,
                        vv,
                        fit.params['I0'],
                        fit.params['x0'],
                        fit.params['y0'],
                        fit.params['sig_x'],
                        fit.params['sig_y'],
                        fit.params['theta'])
    # gauss_2D(uu,vv,*[fit.params[key].value for key in fit.params.keys()])


    fig, ax = plt.subplots(figsize=(13, 7), nrows=1, ncols=2, sharex=True, sharey=True)
    ax[0].scatter(vis.uvw[t, :, 0], vis.uvw[t, :, 1], c=np.abs(data))
    ax[0].imshow(np.abs(fit_data), aspect='auto', origin='lower', extent=[us[0], us[-1], vs[0], vs[-1]])
    ax[0].set_title("Absolute value (amplitude)")
    ax[0].set_xlim([-1000, 1000])
    ax[0].set_ylim([-1000, 1000])
    ax[0].set_xlabel("u")
    ax[0].set_ylabel("v")

    ax[1].scatter(vis.uvw[t, :276, 0], vis.uvw[t, :276, 1], c=np.angle(data[:276]), vmin=-np.pi, vmax=np.pi)
    ax[1].imshow(np.angle(fit_data),
                 aspect='auto',
                 origin='lower',
                 extent=[us[0], us[-1], vs[0], vs[-1]],
                 vmin=-np.pi,
                 vmax=np.pi)
    ax[1].set_title("Phase (position)")
    ax[1].set_xlim([-1000, 1000])
    ax[1].set_ylim([-1000, 1000])
    ax[1].set_xlabel("u")
    ax[1].set_ylabel("v")
    if save:
        plt.savefig(outdir+"visibility_fit_amp_phase_{}MHz_{}.png".format(int(np.round(vis.freq.value)), vis.time.isot[0]))
    if not plot:
        plt.close()
    # uz = np.arange(-500, 500, 1)
    # vz = np.arange(-500, 500, 1)
    # uuz, vvz = np.meshgrid(uz, vz)
    # fit_gaussz = gauss_2D(uuz, vvz, *[fit.params[key].value for key in fit.params.keys()])
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(vis.uvw[t, :276, 0], vis.uvw[t, :276, 1], np.abs(data[:276]), color='r')
    # ax.plot_surface(uuz, vvz, np.abs(fit_gaussz))
    # ax.set_xlim([-500, 500])
    # ax.set_ylim([-500, 500])
    # ax.set_xlabel("u")
    # ax.set_ylabel("v")
    # ax.set_zlabel("Amplitdue (arbitrary)")
    # if save:
    #   plt.savefig("visibility_fit_3d_amp_{}.png".format(vis.time.isot[0]))

    uv_dist = np.sqrt(vis.uvw[t, :, 0] ** 2 + vis.uvw[t, :, 1] ** 2)
    ang_scales = Angle((1 / uv_dist) * u.rad)
    us_p = rotate_coords(us, np.zeros_like(us), fit.params['theta'])
    vs_p = rotate_coords(np.zeros_like(vs), vs, fit.params['theta'])
    fit_data_u = gauss_2D(us_p[0],
                          us_p[1],
                          fit.params['I0'],
                          fit.params['x0'],
                          fit.params['y0'],
                          fit.params['sig_x'],
                          fit.params['sig_y'],
                          fit.params['theta']) #*[fit.params[key].value for key in fit.params.keys()])
    fit_data_v = gauss_2D(vs_p[0],
                          vs_p[1],
                          fit.params['I0'],
                          fit.params['x0'],
                          fit.params['y0'],
                          fit.params['sig_x'],
                          fit.params['sig_y'],
                          fit.params['theta'])#*[fit.params[key].value for key in fit.params.keys()])
    ang_scales_u = Angle((1 / np.sqrt(us_p[0] ** 2 + us_p[1] ** 2)) * u.rad)
    ang_scales_v = Angle((1 / np.sqrt(vs_p[0] ** 2 + vs_p[1] ** 2)) * u.rad)

    plt.figure()
    plt.plot(ang_scales.arcmin, np.abs(data), 'o')
    plt.plot(ang_scales_u.arcmin, np.abs(fit_data_u), color='r')
    plt.plot(ang_scales_v.arcmin, np.abs(fit_data_v), color='r')
    # plt.axvline(sig_to_fwhm(fit.params['sig_x']), color='r')
    # plt.axvline(sig_to_fwhm(fit.params['sig_y']), color='r')
    plt.title('Amplitude vs Angular Scale')
    plt.ylabel('Amplitude (arbitrary)')
    plt.xlabel('Angular scale (arcmin)')
    plt.xscale('log')
    if save:
        plt.savefig(outdir+"visibility_fit_amp_angscale_{}MHz_{}.png".format(int(np.round(vis.freq.value)), vis.time.isot[0]))
    if not plot:
        plt.close()
    fit_vals = [fit.params['I0'].value,
                fit.params['x0'].value,
                fit.params['y0'].value,
                fit.params['sig_x'].value,
                fit.params['sig_y'].value
                # fit.params['theta'].value
                # fit.params['C'].value
                # fit.params ['__lnsigma'].value
                ]
    labels = fit.var_names #['I0', 'x0', 'y0', 'sig_x', 'sig_y']#, 'theta']  # , 'lnsigma']
    val_dict = fit.params.valuesdict()
    corner.corner(fit.flatchain, labels=labels)  # , truths=truths)
    if save:
        plt.savefig(outdir+"visibility_fit_corner_{}MHz_{}.png".format(int(np.round(vis.freq.value)), vis.time.isot[0]))
    if not plot:
        plt.close()
    fig, ax = plt.subplots(fit.nvarys, sharex=True, figsize=(8,7))
    for i in range(fit.nvarys):
        ax[i].plot(fit.chain[:, :, i], 'k', alpha=0.3)
        # ax[i].hlines(truths[i], 0, fit.chain.shape[0], colors='r', zorder=100)
        ax[i].hlines(val_dict[fit.var_names[i]], 0, fit.chain.shape[0], colors='cyan', zorder=100)
        ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("Step Number")
    if save:
        plt.savefig(outdir+"visibility_fit_walkers_{}MHz_{}.png".format(int(np.round(vis.freq.value)), vis.time.isot[0]))
    if not plot:
        plt.close()
    return

def recreate_map(vis, fit, pix=1024, scale=5*u.arcsec, t=0):
    #define grid onto which to put the fitted data
    x = np.arange(pix) * scale
    y = np.arange(pix) * scale
    x = x - x[-1] / 2
    y = y - y[-1] / 2
    x = -x #important!! RA is backwards. Is it always backwards?
    x = x + vis.phase_dir.ra
    y = y + vis.phase_dir.dec
    xx, yy = np.meshgrid(x, y)
    # g_centre = SkyCoord(vis.phase_dir.ra + fit.params['x0'] * u.rad,
    #                     vis.phase_dir.dec + fit.params['y0'] * u.rad,
    #                     distance=vis.phase_dir.distance,
    #                     obstime=Time(vis.time.isot[0]))
    burst_centre_coord = SkyCoord(vis.phase_dir.ra - fit.params['x0'] * u.rad, vis.phase_dir.dec + fit.params['y0'] * u.rad,
                              frame='gcrs',
                              obstime=vis.time.isot[t],
                              obsgeoloc=vis.lofar_gcrs[t].cartesian,
                              obsgeovel=vis.lofar_gcrs[t].velocity.to_cartesian(),
                              distance=vis.lofar_gcrs[t].hcrs.distance,
                              equinox='J2000')
    data = gauss_2D_real((xx.value, yy.value),
                         fit.params['I0'],
                         burst_centre_coord.ra.arcsec,
                         burst_centre_coord.dec.arcsec,
                         Angle(fit.params['sig_x']*u.rad).arcsec,
                         Angle(fit.params['sig_y']*u.rad).arcsec,
                         fit.params['theta'], 0)
    data = data.reshape(pix, pix)

    reference_coord = vis.phase_dir[t]
    reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=vis.lofar_gcrs[t]))
    map_header = sunpy.map.make_fitswcs_header(data,
                                               reference_coord_arcsec,
                                               reference_pixel=u.Quantity([pix/2, pix/2]*u.pixel),
                                               scale=u.Quantity([scale, scale]*u.arcsec/u.pix),
                                               rotation_angle=-sun.P(vis.time.isot),
                                               wavelength=vis.freq,
                                               observatory='LOFAR')
    rec_map = sunpy.map.Map(data, map_header)

    return rec_map

def fit_burst(vis, i):
    """
    Fits burst size position over time
    """
    stokesi = vis.stokes('I', i)
    weights = vis.weight[i, :, 0] + vis.weight[i, :, 3]
    # all this copied from main(), probably should do something about that
    R_av_ang_asec = Angle(R_av_ang.value * u.arcsec)
    best_stokesi = stokesi[np.argwhere((1/weights) < (3 * np.std(1/weights) + np.mean(1/weights)))]
    best_uvw = vis.uvw[i, np.argwhere((1/weights) < (3 * np.std(1/weights) + np.mean(1/weights)))]

    best_uvw = best_uvw.squeeze()
    best_stokesi = best_stokesi.squeeze()

    # Make guess for starting values
    init_params = {"I0": np.max(np.abs(best_stokesi)),
                   "x0": Angle(-1800 * u.arcsec).rad, #-0.00769764,#
                   "y0": Angle(60 * u.arcsec).rad, #0.00412132,#
                   "sig_x": Angle(5 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2))),
                   "sig_y": Angle(7 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2))),
                   "theta": 0}

    params = Parameters()
    params.add_many(("I0", init_params["I0"], True, 0 * init_params["I0"], 2 * init_params["I0"]),
                    ("x0", init_params["x0"], False, - 2 * R_av_ang_asec.rad, 2 * R_av_ang_asec.rad),
                    ("y0", init_params["y0"], False, - 2 * R_av_ang_asec.rad, 2 * R_av_ang_asec.rad),
                    ("sig_x", init_params["sig_x"], True, Angle(5*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))),
                     Angle(30*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2)))))
    params.add("delta", init_params["sig_y"], min=0,
                     max=Angle(30*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))))
    params.add("sig_y", init_params["sig_y"], min=Angle(5*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))),
                     max=Angle(30*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))), expr="delta + sig_x")
    params.add("theta", init_params["theta"],True, -np.pi/2, np.pi/2)

    # Fit amplitude Levenberg–Marquardt algorithm
    # Only vary I0, sig_x, sig_y and theta

    fit_amp = minimize(residual, params,
                       args=(vis.uvw[i, :, 0], vis.uvw[i, :, 1], stokesi, weights, "amplitude"))
    fit_amp.params['I0'].vary = False
    fit_amp.params['x0'].vary = True
    fit_amp.params['y0'].vary = True
    fit_amp.params['sig_x'].vary = False
    fit_amp.params['delta'].vary = False
    fit_amp.params['sig_y'].vary = False
    fit_amp.params['theta'].vary = False
    # fit_amp.params['C'].vary = False

    # uvw[:275] = core baselines
    # Fit phase Levenberg–Marquardt algorithm
    # Only vary x0 and y0
    fit_phase = minimize(residual, fit_amp.params,
                         args=(vis.uvw[i, :275, 0], vis.uvw[i, :275, 1], stokesi[:275], None, "phase"))

    fit_phase.params['I0'].vary = True
    fit_phase.params['x0'].vary = True
    fit_phase.params['y0'].vary = True
    fit_phase.params['sig_x'].vary = True
    fit_phase.params['delta'].vary= True
    fit_phase.params['sig_y'].vary = True
    fit_phase.params['theta'].vary = True
    fit_phase.params.add('__lnsigma', value=np.log(np.std(stokesi)))
    # Fit everything MCMC
    # Make a ball around parameters determined above

    nwalkers = 200
    walker_init_pos = np.array((fit_phase.params['I0'].value,
                                fit_phase.params['x0'].value,
                                fit_phase.params['y0'].value,
                                fit_phase.params['sig_x'].value,
                                fit_phase.params['delta'].value,
                                fit_phase.params['theta'].value,
                                fit_phase.params['__lnsigma'].value
                                )) * (1 + (1e-4 * np.random.randn(nwalkers, len(fit_phase.params)-1)))

    print("Fitting for {}".format(vis.time.isot[i]))
    # pdb.set_trace()

    fit = minimize(residual, fit_phase.params, method='emcee', pos=walker_init_pos,
                   steps=2000, burn=500, nwalkers=nwalkers, progress=True, is_weighted=False,
                   args=(best_uvw[:, 0],
                         best_uvw[:, 1], best_stokesi, None, "all"))

    burst_centre_coord = SkyCoord(vis.phase_dir.ra - fit.params['x0'] * u.rad,
                                  vis.phase_dir.dec + fit.params['y0'] * u.rad,
                                  frame='gcrs',
                                  obstime=vis.time.isot[i],
                                  obsgeoloc=vis.lofar_gcrs[i].cartesian,
                                  obsgeovel=vis.lofar_gcrs[i].velocity.to_cartesian(),
                                  distance=vis.lofar_gcrs[i].hcrs.distance,
                                  equinox='J2000')
    burst_centre_coord_asec = burst_centre_coord.transform_to(frames.Helioprojective(observer=vis.lofar_gcrs[i]))
    return fit, vis.time.isot[i], burst_centre_coord_asec

def main(msin, trange, plot=False):
    """
    Main function. Fits gaussian amplitude using Levenberg–Marquardt algorithm followed by MCMC.
    Uses output of MCMC fit and similarly fits gaussian phase.
    Inputs: msin = Measurement set with LOFAR data
            trange = sunpy.time.TimeRange. Time of burst you want to fit
            plot = boolean. True to show plots of fit default = False
    """
    t0 = time.time()
    vis = LOFAR_vis(msin, trange)
    # Experimenting with subtracting an assumed quiet sun
    # q_sun = LOFAR_vis(msin, TimeRange(vis.time.isot[0][:13]+":45:45", vis.time.isot[0][:13]+":47:15"))
    # q_sun_mean = np.mean([q_sun.data[i * 630:(i + 1) * 630] for i in range(q_sun.data.shape[0] // 630)], axis=0)
    # q_sun_weight = np.mean([q_sun.weight[i * 630:(i + 1) * 630] for i in range(q_sun.weight.shape[0] // 630)], axis=0)
    # diff_vis = (vis.data - q_sun_mean)
    # gauss0 = diff_vis[:,0] + diff_vis[:,3]
    # gauss0 = vis.model[:,0] + vis.model[:,3]
    # gauss0 = vis.uncal[:,0] + vis.uncal[:,3]


    gauss0 = vis.stokes('I')
    weights = vis.weight[0, :,0] + vis.weight[0,:,3]# + q_sun_weight[:,0] + q_sun_weight[:, 3]

    # Adding random noise, normalising the data to see if there's a difference
    # gauss0 = gauss0 + (np.mean(gauss0) * np.random.randn(len(weights)))
    # gauss0 = gauss0/np.max(gauss0)
    # Actual values for model gaussian
    sig_x = fwhm_to_sig(Angle(8 * u.arcmin))
    sig_y = fwhm_to_sig(Angle(12 * u.arcmin))
    x0 = Angle(1000 * u.arcsec)
    y0 = Angle(750 * u.arcsec)
    rot_ang = Angle(10 * u.deg).rad
    R_av_ang_asec = Angle(R_av_ang.value * u.arcsec)
    # pdb.set_trace()
    good_weights = weights[np.argwhere((1/weights) < (3 * np.std(1/weights) + np.mean(1/weights)))]
    good_gauss0 = gauss0[np.argwhere((1/weights) < (3 * np.std(1/weights) + np.mean(1/weights)))]
    # Make guess for starting values
    init_params = {"I0": np.max(np.abs(good_gauss0)),
                   "x0": Angle(-1800 * u.arcsec).rad, #-0.00769764,#
                   "y0": Angle(60 * u.arcsec).rad, #0.00412132,#
                   "sig_x": Angle(5 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2))),
                   "sig_y": Angle(7 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2))),
                   "theta": -np.pi/4}
                   # "C": 0}

    params = Parameters()
    params.add_many(("I0", init_params["I0"], True, 0 * init_params["I0"], 2 * init_params["I0"]),
                    ("x0", init_params["x0"], False, - 2 * R_av_ang_asec.rad, 2 * R_av_ang_asec.rad),
                    ("y0", init_params["y0"], False, - 2 * R_av_ang_asec.rad, 2 * R_av_ang_asec.rad),
                    ("sig_x", init_params["sig_x"], True, Angle(5*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))),
                     Angle(30*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2)))))
    params.add("delta", init_params["sig_y"], min=0,
                     max=Angle(30*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))))
    params.add("sig_y", init_params["sig_y"], min=Angle(5*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))),
                     max=Angle(30*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))), expr="sig_x + delta")
    params.add("theta", init_params["theta"],True, -np.pi/2, np.pi/2)
                    # ("C", init_params["C"], False, 0, 0.5 * np.max(np.abs(gauss0))))

    # Experimenting with different errors/weights
    # error = np.abs(gauss0 * np.sqrt((vis.weight[:, 0]/vis.data[:,0]) ** 2 +
    #                                 (vis.weight[:, 3]/vis.data[:,3]) ** 2))# +
                                    # (q_sun_weight[:,0]/q_sun_mean[:,0]) ** 2 +
                                    # (q_sun_weight[:, 3]/q_sun_mean[:, 3]) ** 2))


                                    # (q_sun.weight[:, 0]/q_sun.data[:,0]) ** 2 +
                                    # (q_sun.weight[:, 3]/q_sun.data[:,3]) ** 2)

    # error = 1/error
    uv_dist = np.sqrt(vis.uvw[0,:630, 0] ** 2 + vis.uvw[0,:630, 1] ** 2)
    ang_scales = Angle((1 / uv_dist) * u.rad)
    error = weights * np.std(gauss0)
    # error = ang_scales.rad
    # error = briggs(weights, -1)
    # Fit amplitude Levenberg–Marquardt algorithm
    # Only vary I0, sig_x, sig_y and theta

    fit_amp = minimize(residual, params,
                       args=(vis.uvw[0,:, 0], vis.uvw[0,:, 1], gauss0, weights, "amplitude"))
    fit_amp.params['I0'].vary = False
    fit_amp.params['x0'].vary = True
    fit_amp.params['y0'].vary = True
    fit_amp.params['sig_x'].vary = False
    fit_amp.params['delta'].vary = False
    fit_amp.params['sig_y'].vary = False
    fit_amp.params['theta'].vary = False
    # fit_amp.params['C'].vary = False

    # uvw[:276] = core baselines
    # Fit phase Levenberg–Marquardt algorithm
    # Only vary x0 and y0
    # pdb.set_trace()
    fit_phase = minimize(residual, fit_amp.params,
                         args=(vis.uvw[0,:275, 0], vis.uvw[0, :275, 1], gauss0[:275], None, "phase"))
    # assume first guess is correct and update min, max parameter values
    # don't do this because the first guess is rarely correct
    # fit_phase.params['x0'].min = fit_phase.params['x0'] - (R_av_ang_asec.rad / 2)
    # fit_phase.params['x0'].max = fit_phase.params['x0'] + (R_av_ang_asec.rad / 2)
    # fit_phase.params['y0'].min = fit_phase.params['y0'] - (R_av_ang_asec.rad / 2)
    # fit_phase.params['y0'].max = fit_phase.params['y0'] + (R_av_ang_asec.rad / 2)
    # fit_phase.params['sig_x'].min = fit_phase.params['sig_x'] - Angle(5 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2)))
    # fit_phase.params['sig_x'].max = fit_phase.params['sig_x'] + Angle(5 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2)))
    # fit_phase.params['sig_y'].min = fit_phase.params['sig_y'] - Angle(5 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2)))
    # fit_phase.params['sig_y'].max = fit_phase.params['sig_y'] + Angle(5 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2)))

    fit_phase.params['I0'].vary = True
    fit_phase.params['x0'].vary = True
    fit_phase.params['y0'].vary = True
    fit_phase.params['sig_x'].vary = True
    fit_phase.params['delta'].vary= True
    fit_phase.params['sig_y'].vary = True
    fit_phase.params['theta'].vary = True
    # fit_phase.params['C'].vary = False
    fit_phase.params.add('__lnsigma', value=np.log(np.std(gauss0)))#, min=np.log(0.001), max=np.log(2))
    # Fit everything MCMC
    # Make a ball around parameters determined above
    # pdb.set_trace()
    nwalkers = 200
    walker_init_pos = np.array((fit_phase.params['I0'].value,
                                fit_phase.params['x0'].value,
                                fit_phase.params['y0'].value,
                                fit_phase.params['sig_x'].value,
                                fit_phase.params['delta'].value,
                                # fit_phase.params['sig_y'].value,
                                fit_phase.params['theta'].value,
                                # fit_phase.params['C'].value
                                fit_phase.params['__lnsigma'].value
                                )) * (1 + (1e-4 * np.random.randn(nwalkers, len(fit_phase.params)-1)))

    print("Fitting for {}".format(vis.time.isot))
    # pdb.set_trace()

    fit = minimize(residual, fit_phase.params, method='emcee', pos=walker_init_pos,
                   steps=2000, burn=500, nwalkers=nwalkers, progress=True, is_weighted=False,
                   args=(vis.uvw[0,np.argwhere((1/weights) < (3 * np.std(1/weights) + np.mean(1/weights))), 0],
                         vis.uvw[0,np.argwhere((1/weights) < (3 * np.std(1/weights) + np.mean(1/weights))), 1], good_gauss0, None, "all"))

    # outdir = "vis_fits/51MHz/"+vis.time.isot[0][:10].replace('-','_') + "/"
    outdir = "vis_fits/30MHz/"+vis.time.isot[0][:10].replace('-','_') + "/"
    plot_fit(vis, gauss0, fit, plot, save=False, outdir=outdir)

    print("Time to run {}".format(time.time() - t0))
    burst_centre_coord = SkyCoord(vis.phase_dir.ra - fit.params['x0'] * u.rad,
                                  vis.phase_dir.dec + fit.params['y0'] * u.rad,
                                  frame='gcrs',
                                  obstime=vis.time.isot[0],
                                  obsgeoloc=vis.lofar_gcrs.cartesian,
                                  obsgeovel=vis.lofar_gcrs.velocity.to_cartesian(),
                                  distance=vis.lofar_gcrs.hcrs.distance,
                                  equinox='J2000')
    burst_centre_coord_asec = burst_centre_coord.transform_to(frames.Helioprojective(observer=vis.lofar_gcrs))
    return fit, vis.time.isot[0], burst_centre_coord_asec#, fit_amp, fit_phase


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
            trange = TimeRange(trange[0], trange[1])  # ("2019-04-04T14:08:00", "2019-04-04T14:17:00")
        elif len(trange) == 1:
            tstart = Time(trange[0])
            trange = TimeRange(tstart, tstart + 0.16 * u.s)

        fit, burst_time, burst_centre_coord = main(msin, trange, plot=True)
        # vis = LOFAR_vis(msin, trange)
        # with Pool() as pool:
        #     fit_burst_time_dir = np.array(pool.starmap(fit_burst, product([vis], range(len(vis.time)))))
        # fit_burst_time_dir = fit_burst_time_dir.T
        # fit, burst_time, burst_centre_coord = fit_burst_time_dir
        # fit_df = pd.DataFrame(fit, index=burst_time)
        # fit_df['burst_centre_coord'] = burst_centre_coord
        # I0s = [fit.params['I0'].value for fit in fit_df[0]]
        # #specific to one burst from here on out
        # time_from_start = (vis.time - vis.time[0]).sec
        # t_start = 4.5
        # t_end = 7
        # plt.figure()
        # plt.plot(time_from_start, I0s, 'o')
        # plt.xlabel("Time (s)")
        # plt.ylabel("Peak Intensity")
        # plt.axvline(t_start, c='r')
        # plt.axvline(t_end, c='r')
        #
        # sig_xs = np.array([fit.params['sig_x'].value for fit in fit_df[0]])
        # sig_ys = np.array([fit.params['sig_y'].value for fit in fit_df[0]])
        # areas = Angle(sig_xs * u.rad) * Angle(sig_ys * u.rad) * np.pi
        #
        # sig_x_std = np.array([fit.params['sig_x'].stderr for fit in fit_df[0]])
        # deltas_std = np.array([fit.params['delta'].stderr for fit in fit_df[0]])
        # sig_y_std = np.sqrt(sig_x_std**2 + deltas_std**2)
        # area_std = areas * np.sqrt((sig_x_std / sig_xs) ** 2 + (sig_y_std / sig_ys) ** 2)
        #
        # areas_amin = areas.to(u.arcmin ** 2)
        # area_std_amin = area_std.to(u.arcmin ** 2)
        #
        # b_start = np.argwhere(time_from_start > t_start)[0][0]
        # b_end = np.argwhere(time_from_start < t_end)[-1][0]
        #
        # area_growth = LinearModel()
        # pars = area_growth.guess(areas_amin[b_start:b_end], x=time_from_start[b_start:b_end])
        # area_growth_fit = area_growth.fit(areas_amin[b_start:b_end].value, pars, x=time_from_start[b_start:b_end], weights=1/area_std_amin[b_start:b_end].value)
        #
        # plt.figure()
        # plt.errorbar(time_from_start[b_start:b_end], areas_amin[b_start:b_end].value,
        #              area_std_amin[b_start:b_end].value, ls='', marker='o')
        # plt.plot(time_from_start[b_start:b_end], area_growth_fit.best_fit)
        # plt.xlabel("Time (s)")
        # plt.ylabel(r"Area (arcmin$^2$)")
        # rec_map = recreate_map(vis, fit)

        # g_centre = SkyCoord(phase_dir.ra + fit.params['x0'] * u.rad, phase_dir.dec + fit.params['y0'] * u.rad,
        #                     distance=phase_dir.distance, obstime=burst_time)
        # report_fit(fit)
        # print("Aspect Ratio: {}".format(fit.params['sig_x'].value / fit.params['sig_y'].value))
        plt.show()
    else:
        df = pd.read_pickle(pickle)
        trange_list = []
        for i, t in enumerate(df[df.columns[0]]):
            tstart = Time(t)
            trange = TimeRange(tstart, tstart + 0.16 * u.s)
            trange_list.append(trange)
        # with Pool() as pool:
        #     fit_burst_time_dir = np.array(pool.starmap(main, product([msin], trange_list)))
        fit_burst_time_dir = []
        for i in range(len(trange_list)):
            fit_burst_time = main(msin, trange_list[i])
            fit_burst_time_dir.append(fit_burst_time)
        print("fitting and plotting finished")
        fit_burst_time_dir = np.array(fit_burst_time_dir)
        fit_burst_time_dir = fit_burst_time_dir.T
        fit, burst_time, burst_centre_coord = fit_burst_time_dir
        fit_df = pd.DataFrame([f.params.valuesdict() for f in fit], index=burst_time)
        fit_df['redchi'] = [f.redchi for f in fit]
        fit_df['burst_centre_coord'] = burst_centre_coord
        std_dict = {key + '_stderr': None for key in fit[0].params.keys()}
        for key in fit[0].params.keys():
            std_dict[key + '_stderr'] = [f.params[key].stderr for f in fit]
        fit_std_df = pd.DataFrame(std_dict, index=burst_time)
        fit_df = pd.concat([fit_df, fit_std_df], axis='columns')
        pickle_path = "burst_properties_30MHz_good_times_visibility_fit_{}.pkl".format(trange.start.isot[:10])
        # pickle_path = "burst_properties_51MHz_visibility_fit_{}.pkl".format(trange.start.isot[:10])
        fit_df.to_pickle(pickle_path)
