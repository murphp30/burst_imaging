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

from itertools import product

from astropy.constants import au, e, eps0, c, m_e, R_sun
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time, TimeDelta
from casacore import tables
from lmfit import Model, Parameters, minimize, report_fit
from matplotlib import dates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from sunpy.sun.constants import average_angular_size as R_av_ang
from sunpy.time import TimeRange

sys.path.insert(1, '/mnt/LOFAR-PSP/pearse_2ndperihelion/scripts')
sys.path.insert(1, '/Users/murphp30/mnt/LOFAR-PSP/pearse_2ndperihelion/scripts')
plt.style.use('seaborn-colorblind')


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


def gauss_2D(u, v, I0, x0, y0, sig_x, sig_y, theta, C):
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
    pa = -theta  # reverse gauss position angle because we're rotating the gaussian to align with our coordinates
    # it doesn't work if you don't so that's reason enough...

    u_p, v_p = rotate_coords(u, v, pa)
    x0_p, y0_p = rotate_coords(x0, y0, pa)
    amp = I0 #/ (2 * np.pi)

    shift = np.exp(-2 * np.pi * 1j * (u * x0 + v * -y0))
    V = shift * (amp * np.exp(
        -((((sig_x ** 2) * ((2 * np.pi * u_p) ** 2)) / 2) + (((sig_y ** 2) * ((2 * np.pi * v_p) ** 2)) / 2))) + C)
    return V


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

    if fit == "amplitude":
        model = gauss_2D(u.value, v.value,
                         params_dict['I0'], 0, 0,
                         params_dict['sig_x'], params_dict['sig_y'], params_dict['theta'], 0)
        if weights is None:
            resid = np.abs(data) - np.abs(model)
        else:
            resid = (np.abs(data) - np.abs(model)) * weights
    elif fit == "phase":
        model = gauss_2D(u.value, v.value,
                         params_dict['I0'], params_dict['x0'], params_dict['y0'],
                         params_dict['sig_x'], params_dict['sig_y'], params_dict['theta'], 0)
        if weights is None:
            resid = np.angle(data) - np.angle(model)
        else:
            resid = (np.angle(data) - np.angle(model)) * weights
    elif fit == "all":
        model = gauss_2D(u.value, v.value,
                         params_dict['I0'], params_dict['x0'], params_dict['y0'],
                         params_dict['sig_x'], params_dict['sig_y'], params_dict['theta'], 0)
        if weights is None:
            resid = np.abs(data - model)#np.sqrt((np.abs(data - model))**2 + (np.angle(data - model))**2)
        else:
            resid = (np.abs(data - model) + np.angle(data - model))* weights
            #np.abs((data - model)) * weights
                # np.sqrt((np.abs(data - model))**2 + (np.angle(data - model))**2)*weights
            #np.sqrt((data.real - model.real)**2 + (data.imag - model.imag)**2) * weights#np.abs((data - model)) * weights
    else:
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
        with tables.table(self.fname + 'FIELD', ack=False) as tfield:
            phase_dir = tfield.col('PHASE_DIR')[0][0]
            self.phase_dir = SkyCoord(*phase_dir * u.rad)
        with tables.table(self.fname + 'SPECTRAL_WINDOW', ack=False) as tspec:
            freq = tspec.col('REF_FREQUENCY')[0] * u.Hz
            self.wlen = (c / freq).decompose()
            self.freq = freq.to(u.MHz)
        self.__get_data()

    def __get_data(self):
        with tables.table(self.fname, ack=False) as t:
            self.dt = t.col('INTERVAL')[0] * u.s
            ts = self.trange.start.mjd * 24 * 3600
            te = self.trange.end.mjd * 24 * 3600

            with tables.taql('SELECT * FROM $t WHERE TIME > $ts AND TIME < $te') as t1:
                antenna1 = t1.getcol('ANTENNA1')
                antenna2 = t1.getcol('ANTENNA2')
                cross_cors = np.where(antenna1 != antenna2)
                self.flag = t1.getcol('FLAG')[cross_cors]
                self.antenna1 = antenna1[cross_cors]
                self.antenna2 = antenna2[cross_cors]
                self.uvw = (t1.getcol('UVW')[cross_cors]) * u.m / self.wlen
                self.time = Time(t1.getcol('TIME', rowincr=int(self.nbaselines)) / 24 / 3600, format='mjd')
                self.data = t1.getcol('CORRECTED_DATA')[:, 0, :][cross_cors]
                self.uncal = t1.getcol('DATA')[:, 0, :][cross_cors]
                self.model = t1.getcol('MODEL_DATA')[:, 0, :][cross_cors]
                self.weight = t1.getcol('WEIGHT_SPECTRUM')[:, 0, :][cross_cors]

    def stokes(self, param):
        """
        Returns inputed Stokes Parameter of either I, Q, U or V (upper case only)
        """
        accepted_params = ['I', 'Q', 'U', 'V']
        if param not in accepted_params:
            print("Please choose one of: I, Q, U, V.")
            return
        if param == 'I':
            return self.data[:, 0] + self.data[:, 3]
        elif param == 'Q':
            return self.data[:, 0] - self.data[:, 3]
        elif param == 'U':
            return np.real(self.data[:, 1] + self.data[:, 2])
        elif param == 'V':
            return np.imag(self.data[:, 1] - self.data[:, 2])

    def plot(self):
        """
        I don't remember why I wrote this
        Outputs a plot of the uv coverage for the MS
        """
        plt.scatter(self.uvw[:, 0], self.uvw[:, 1])

def briggs(w_i, R):
    """
    Hacky implementation of Briggs robustness weighting
    https://casa.nrao.edu/Release4.1.0/doc/UserMan/UserMansu262.html
    """
    W_k = (np.max(w_i)/w_i)
    f_sq = (5*(10**-R))/(np.sum(W_k**2)/np.sum(w_i))
    return w_i/(1+W_k*f_sq)

def plot_fit(vis, data, fit, plot=True):
    """
    Plots data and fit to data, MCMC corner plot and the chain itself
    Inputs: vis = LOFAR_vis object for burst
            data = data used in fit (probably a better way of getting this because it's
            technically part of vis but I didn't want to change things twice while I was messing around with it)
            fit = lmfit.minimizer.MinimizerResult of MCMC fit
            plot = boolean. False runs plt.close() after each plot, default = True/
    """
    us = np.arange(np.min(vis.uvw[:, 0].value), np.max(vis.uvw[:, 0].value), 10)
    vs = np.arange(np.min(vis.uvw[:, 1].value), np.max(vis.uvw[:, 1].value), 10)
    uu, vv = np.meshgrid(us, vs)
    fit_data = gauss_2D(uu, vv, *[fit.params[key].value for key in fit.params.keys()])

    fig, ax = plt.subplots(figsize=(13, 7), nrows=1, ncols=2, sharex=True, sharey=True)
    ax[0].scatter(vis.uvw[:630, 0], vis.uvw[:630, 1], c=np.abs(data))
    ax[0].imshow(np.abs(fit_data), aspect='auto', origin='lower', extent=[us[0], us[-1], vs[0], vs[-1]])
    ax[0].set_title("Absolute value (amplitude)")
    ax[0].set_xlim([-1000, 1000])
    ax[0].set_ylim([-1000, 1000])
    ax[0].set_xlabel("u")
    ax[0].set_ylabel("v")

    ax[1].scatter(vis.uvw[:276, 0], vis.uvw[:276, 1], c=np.angle(data[:276]), vmin=-np.pi, vmax=np.pi)
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
    plt.savefig("visibility_fit_amp_phase_{}.png".format(vis.time.isot[0]))
    if not plot:
        plt.close()
    # uz = np.arange(-500, 500, 1)
    # vz = np.arange(-500, 500, 1)
    # uuz, vvz = np.meshgrid(uz, vz)
    # fit_gaussz = gauss_2D(uuz, vvz, *[fit.params[key].value for key in fit.params.keys()])
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(vis.uvw[:276, 0], vis.uvw[:276, 1], np.abs(data[:276]), color='r')
    # ax.plot_surface(uuz, vvz, np.abs(fit_gaussz))
    # ax.set_xlim([-500, 500])
    # ax.set_ylim([-500, 500])
    # ax.set_xlabel("u")
    # ax.set_ylabel("v")
    # ax.set_zlabel("Amplitdue (arbitrary)")
    # plt.savefig("visibility_fit_3d_amp_{}.png".format(vis.time.isot[0]))

    uv_dist = np.sqrt(vis.uvw[:630, 0] ** 2 + vis.uvw[:630, 1] ** 2)
    ang_scales = Angle((1 / uv_dist) * u.rad)
    us_p = rotate_coords(us, np.zeros_like(us), fit.params['theta'])
    vs_p = rotate_coords(np.zeros_like(vs), vs, fit.params['theta'])
    fit_data_u = gauss_2D(us_p[0], us_p[1], *[fit.params[key].value for key in fit.params.keys()])
    fit_data_v = gauss_2D(vs_p[0], vs_p[1], *[fit.params[key].value for key in fit.params.keys()])
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
    plt.savefig("visibility_fit_amp_angscale_{}.png".format(vis.time.isot[0]))
    if not plot:
        plt.close()
    fit_vals = [fit.params['I0'].value,
                fit.params['x0'].value,
                fit.params['y0'].value,
                fit.params['sig_x'].value,
                fit.params['sig_y'].value,
                fit.params['theta'].value
                # fit.params['C'].value
                # fit.params ['__lnsigma'].value
                ]
    labels = ['I0', 'x0', 'y0', 'sig_x', 'sig_y', 'theta']  # , 'lnsigma']
    corner.corner(fit.flatchain, labels=labels)  # , truths=truths)
    plt.savefig("visibility_fit_corner_{}.png".format(vis.time.isot[0]))
    if not plot:
        plt.close()
    fig, ax = plt.subplots(fit.nvarys, sharex=True, figsize=(8,7))
    for i in range(fit.nvarys):
        ax[i].plot(fit.chain[:, :, i], 'k', alpha=0.3)
        # ax[i].hlines(truths[i], 0, fit.chain.shape[0], colors='r', zorder=100)
        ax[i].hlines(fit_vals[i], 0, fit.chain.shape[0], colors='cyan', zorder=100)
        ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("Step Number")
    plt.savefig("visibility_fit_walkers_{}.png".format(vis.time.isot[0]))
    if not plot:
        plt.close()
    return


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
    gauss0 = vis.stokes('I')
    weights = vis.weight[:,0] + vis.weight[:,3]# + q_sun_weight[:,0] + q_sun_weight[:, 3]
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

    # Make guess for starting values
    init_params = {"I0": np.max(np.abs(gauss0)),
                   "x0": Angle(500 * u.arcsec).rad,
                   "y0": Angle(750 * u.arcsec).rad,
                   "sig_x": Angle(5 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2))),
                   "sig_y": Angle(7 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2))),
                   "theta": 0,
                   "C": 0}

    params = Parameters()
    params.add_many(("I0", init_params["I0"], True, 0.25 * init_params["I0"], 2 * init_params["I0"]),
                    ("x0", init_params["x0"], False, - 2 * R_av_ang_asec.rad, 2 * R_av_ang_asec.rad),
                    ("y0", init_params["y0"], False, - 2 * R_av_ang_asec.rad, 2 * R_av_ang_asec.rad),
                    ("sig_x", init_params["sig_x"], True, Angle(5*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))),
                     Angle(20*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2)))),
                    ("sig_y", init_params["sig_y"], True, Angle(5*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))),
                     Angle(20*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2)))),
                    ("theta", init_params["theta"], True, -np.pi / 8, np.pi / 8),
                    ("C", init_params["C"], False, 0, 0.5 * np.max(np.abs(gauss0))))

    # Experimenting with different errors/weights
    # error = np.abs(gauss0 * np.sqrt((vis.weight[:, 0]/vis.data[:,0]) ** 2 +
    #                                 (vis.weight[:, 3]/vis.data[:,3]) ** 2))# +
                                    # (q_sun_weight[:,0]/q_sun_mean[:,0]) ** 2 +
                                    # (q_sun_weight[:, 3]/q_sun_mean[:, 3]) ** 2))


                                    # (q_sun.weight[:, 0]/q_sun.data[:,0]) ** 2 +
                                    # (q_sun.weight[:, 3]/q_sun.data[:,3]) ** 2)

    # error = 1/error
    error = weights
    # error = briggs(weights, -1)
    # Fit amplitude Levenberg–Marquardt algorithm
    # Only vary I0, sig_x, sig_y and theta
    fit_amp = minimize(residual, params,
                       args=(vis.uvw[:, 0], vis.uvw[:, 1], gauss0, error, "amplitude"))
    fit_amp.params['I0'].vary = False
    fit_amp.params['x0'].vary = True
    fit_amp.params['y0'].vary = True
    fit_amp.params['sig_x'].vary = False
    fit_amp.params['sig_y'].vary = False
    fit_amp.params['theta'].vary = False
    fit_amp.params['C'].vary = False

    # uvw[:276] = core baselines
    # Fit phase Levenberg–Marquardt algorithm
    # Only vary x0 and y0
    fit_phase = minimize(residual, fit_amp.params,
                         args=(vis.uvw[:, 0], vis.uvw[:, 1], gauss0, error, "phase"))
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
    fit_phase.params['sig_y'].vary = True
    fit_phase.params['theta'].vary = True
    fit_phase.params['C'].vary = False
    # fit_phase.params.add('__lnsigma', value=np.log(0.1))#, min=np.log(0.001), max=np.log(2))
    # Fit everything MCMC
    # Make a ball around parameters determined above
    nwalkers = 350
    walker_init_pos = np.array((fit_phase.params['I0'].value,
                                fit_phase.params['x0'].value,
                                fit_phase.params['y0'].value,
                                fit_phase.params['sig_x'].value,
                                fit_phase.params['sig_y'].value,
                                fit_phase.params['theta'].value
                                # fit_phase.params['C'].value
                                # fit_phase.params['__lnsigma'].value
                                )) * (1 + (1e-4 * np.random.randn(nwalkers, len(init_params) - 1)))

    print("Fitting for {}".format(vis.time.isot))
    # pdb.set_trace()
    fit = minimize(residual, fit_phase.params, method='emcee', pos=walker_init_pos,
                   steps=2000, burn=450, nwalkers=nwalkers, progress=True, is_weighted=True,
                   args=(vis.uvw[:, 0], vis.uvw[:, 1], gauss0, error, "all"))

    plot_fit(vis, gauss0, fit, plot)

    print("Time to run {}".format(time.time() - t0))
    return fit, vis.time.isot[0]#, fit_amp, fit_phase


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

        fit, burst_time = main(msin, trange, plot=True)
        report_fit(fit)
        print("Aspect Ratio: {}".format(fit.params['sig_x'].value / fit.params['sig_y'].value))
        plt.show()
    else:
        df = pd.read_pickle(pickle)
        trange_list = []
        for i, t in enumerate(df[df.columns[0]]):
            tstart = Time(t)
            trange = TimeRange(tstart, tstart + 0.16 * u.s)
            trange_list.append(trange)
        with Pool() as pool:
            fit_burst_time = np.array(pool.starmap(main, product([msin], trange_list)))
        print("fitting and plotting finished")
        fit_burst_time = fit_burst_time.T
        fit, burst_time = fit_burst_time
        fit_df = pd.DataFrame([f.params.valuesdict() for f in fit], index=burst_time)
        std_dict = {key + '_stderr': None for key in fit[0].params.keys()}
        for key in fit[0].params.keys():
            std_dict[key + '_stderr'] = [f.params[key].stderr for f in fit]
        fit_std_df = pd.DataFrame(std_dict, index=burst_time)
        fit_df = pd.concat([fit_df, fit_std_df], axis='columns')
        pickle_path = "burst_properties_visibility_fit_{}.pkl".format(trange.start.isot[:10])
        fit_df.to_pickle(pickle_path)
