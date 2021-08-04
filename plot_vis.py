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
    amp = I0 / (2 * np.pi)

    shift = np.exp(-2 * np.pi * 1j * (u * x0 + v * -y0))
    V = shift * (amp * np.exp(
        -((sig_x ** 2 * (2 * np.pi * u_p) ** 2 / 2) + (sig_y ** 2 * (2 * np.pi * v_p) ** 2 / 2))) + C)
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
    model = gauss_2D(u.value, v.value, *[params[key].value for key in params.keys()])
    if fit == "amplitude":
        if weights is None:
            resid = np.abs(data) - np.abs(model)
        else:
            resid = (np.abs(data) - np.abs(model)) * weights
    elif fit == "phase":
        if weights is None:
            resid = np.angle(data) - np.angle(model)
        else:
            resid = (np.angle(data) - np.angle(model)) * weights
    elif fit == "all":
        if weights is None:
            resid = np.abs(data - model)
        else:
            resid = np.abs((data - model) * weights)
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

    def stokes(self, param):
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
        plt.scatter(self.uvw[:, 0], self.uvw[:, 1])


def main(msin, trange):
    """
    Main function. Fits gaussian amplitude using Levenberg–Marquardt algorithm followed by MCMC.
    Uses output of MCMC fit and similarly fits gaussian phase.
    """
    t0 = time.time()
    vis = LOFAR_vis(msin, trange)
    # gauss0 = vis.model[:, 0] + vis.model[:, 3]
    gauss0 = vis.stokes('I')
    us = np.arange(np.min(vis.uvw[:, 0].value), np.max(vis.uvw[:, 0].value), 10)
    vs = np.arange(np.min(vis.uvw[:, 1].value), np.max(vis.uvw[:, 1].value), 10)
    uu, vv = np.meshgrid(us, vs)
    sig_x = fwhm_to_sig(Angle(10 * u.arcmin))
    sig_y = fwhm_to_sig(Angle(15 * u.arcmin))
    x0 = Angle(1000 * u.arcsec)
    y0 = Angle(750 * u.arcsec)
    rot_ang = Angle(15 * u.deg).rad
    R_av_ang_asec = Angle(R_av_ang.value * u.arcsec)

    # determine starting values
    init_params = {"I0": np.max(np.abs(gauss0)),
                   "x0": Angle(-750 * u.arcsec).rad,
                   "y0": Angle(1000 * u.arcsec).rad,
                   "sig_x": Angle(8 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2))),
                   "sig_y": Angle(17 * u.arcmin).rad / (2 * np.sqrt(2 * np.log(2))),
                   "theta": 0,
                   "C": 1e-2}

    params = Parameters()
    params.add_many(("I0", init_params["I0"], True, 0.75 * init_params["I0"], None),
                    ("x0", init_params["x0"], False, - 2 * R_av_ang_asec.rad, 2 * R_av_ang_asec.rad),
                    ("y0", init_params["y0"], False, - 2 * R_av_ang_asec.rad, 2 * R_av_ang_asec.rad),
                    ("sig_x", init_params["sig_x"], True, 0, 2 * init_params["sig_x"]),
                    ("sig_y", init_params["sig_y"], True, 0, 2 * init_params["sig_y"]),
                    ("theta", init_params["theta"], False, -np.pi / 8, np.pi / 8),
                    ("C", init_params["C"], True, 0.0 * np.min(np.abs(gauss0)), np.max(np.abs(gauss0))))

    error_amp = np.ones_like(np.abs(gauss0)) * np.std(np.abs(gauss0))
    # Fit amplitude Levenberg–Marquardt algorithm
    fit_amp = minimize(residual, params, args=(vis.uvw[:, 0], vis.uvw[:, 1], gauss0, 1 / error_amp, "amplitude"))

    # Fit amplitude MCMC
    # nwalkers = 200
    # walker_init_pos = np.array((fit_amp_lm.params['I0'].value,
    #                             fit_amp_lm.params['sig_x'].value,
    #                             fit_amp_lm.params['sig_y'].value,
    #                             fit_amp_lm.params['theta'].value,
    #                             fit_amp_lm.params['C'].value
    #                             )) + (1e-4 * np.random.randn(nwalkers, fit_amp_lm.nvarys))
    # fit_amp = minimize(residual, fit_amp_lm.params, method = 'emcee', is_weighted = True, pos = walker_init_pos,
    #                    steps = 2500 , burn = 150, nwalkers = nwalkers,
    #                    args=(vis.uvw[:, 0], vis.uvw[:, 1], gauss0, 1 / error_amp, "amplitude"))
    fit_amp.params['I0'].vary = False
    fit_amp.params['x0'].vary = True
    fit_amp.params['y0'].vary = True
    fit_amp.params['sig_x'].vary = False
    fit_amp.params['sig_y'].vary = False
    fit_amp.params['theta'].vary = False
    fit_amp.params['C'].vary = False

    error_phase = np.ones_like(np.angle(gauss0)) * np.std(np.angle(gauss0))
    # uvw[:276] because only the core baselines look nice
    # Fit phase Levenberg–Marquardt algorithm
    fit_phase = minimize(residual, fit_amp.params,
                         args=(vis.uvw[:276, 0], vis.uvw[:276, 1], gauss0[:276], None, "phase"))
    fit_phase.params['x0'].min = fit_phase.params['x0'] - (R_av_ang_asec.rad / 2)
    fit_phase.params['x0'].max = fit_phase.params['x0'] + (R_av_ang_asec.rad / 2)
    fit_phase.params['y0'].min = fit_phase.params['y0'] - (R_av_ang_asec.rad / 2)
    fit_phase.params['y0'].max = fit_phase.params['y0'] + (R_av_ang_asec.rad / 2)
    # Fit phase MCMC
    # nwalkers = 50
    # walker_init_pos = np.array((fit_phase.params['x0'].value,
    #                             fit_phase.params['y0'].value)) + (1e-4 * np.random.randn(nwalkers, fit_phase.nvarys))
    # fit_phase_mc = minimize(residual, fit_phase.params, method = 'emcee', pos = walker_init_pos,
    #                steps = 2500, burn = 75, nwalkers = nwalkers, progress = True,
    #                args=(vis.uvw[:276,0], vis.uvw[:276,1], gauss0[:276], None, "phase" ))

    fit_phase.params['I0'].vary = True
    fit_phase.params['x0'].vary = True
    fit_phase.params['y0'].vary = True
    fit_phase.params['sig_x'].vary = True
    fit_phase.params['sig_y'].vary = True
    fit_phase.params['theta'].vary = False 
    fit_phase.params['C'].vary = True

    nwalkers = 350
    walker_init_pos = np.array((fit_phase.params['I0'].value,
                                fit_phase.params['x0'].value,
                                fit_phase.params['y0'].value,
                                fit_phase.params['sig_x'].value,
                                fit_phase.params['sig_y'].value,
                                #fit_phase.params['theta'].value,
                                fit_phase.params['C'].value
                                )) + (1e-4 * np.random.randn(nwalkers, len(init_params)-1))
    error = np.ones_like(gauss0) * np.std(gauss0)
    fit = minimize(residual, fit_phase.params, method='emcee', pos=walker_init_pos,
                   steps=3000, burn=300, nwalkers=nwalkers, progress=True, is_weighted=True,
                   args=(vis.uvw[:, 0], vis.uvw[:, 1], gauss0, 1 / error, "all"))

    fit_gauss0 = gauss_2D(uu, vv, *[fit.params[key].value for key in fit.params.keys()])

    fig, ax = plt.subplots(figsize=(13, 7), nrows=1, ncols=2, sharex=True, sharey=True)
    ax[0].scatter(vis.uvw[:630, 0], vis.uvw[:630, 1], c=np.abs(gauss0))
    ax[0].imshow(np.abs(fit_gauss0), aspect='auto', origin='lower', extent=[us[0], us[-1], vs[0], vs[-1]])
    ax[0].set_title("Absolute value (amplitude)")
    ax[0].set_xlim([-1000, 1000])
    ax[0].set_ylim([-1000, 1000])
    ax[0].set_xlabel("u")
    ax[0].set_ylabel("v")

    ax[1].scatter(vis.uvw[:276, 0], vis.uvw[:276, 1], c=np.angle(gauss0[:276]), vmin=-np.pi, vmax=np.pi)
    ax[1].imshow(np.angle(fit_gauss0),
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

    uz = np.arange(-500, 500, 1)
    vz = np.arange(-500, 500, 1)
    uuz, vvz = np.meshgrid(uz, vz)
    fit_gaussz = gauss_2D(uuz, vvz, *[fit.params[key].value for key in fit.params.keys()])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(vis.uvw[:276, 0], vis.uvw[:276, 1], np.abs(gauss0[:276]), color='r')
    ax.plot_surface(uuz, vvz, np.abs(fit_gaussz))
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_zlabel("Amplitdue (arbitrary)")
    plt.savefig("visibility_fit_3d_amp_{}.png".format(vis.time.isot[0]))

    uv_dist = np.sqrt(vis.uvw[:630, 0] ** 2 + vis.uvw[:630, 1] ** 2)
    ang_scales = Angle((1 / uv_dist) * u.rad)
    us_p = rotate_coords(us, np.zeros_like(us), fit.params['theta'])
    vs_p = rotate_coords(np.zeros_like(vs), vs, fit.params['theta'])
    fit_gauss_u = gauss_2D(us_p[0], us_p[1], *[fit.params[key].value for key in fit.params.keys()])
    fit_gauss_v = gauss_2D(vs_p[0], vs_p[1], *[fit.params[key].value for key in fit.params.keys()])
    ang_scales_u = Angle((1 / np.sqrt(us_p[0] ** 2 + us_p[1] ** 2)) * u.rad)
    ang_scales_v = Angle((1 / np.sqrt(vs_p[0] ** 2 + vs_p[1] ** 2)) * u.rad)

    plt.figure()
    plt.plot(ang_scales.arcmin, np.abs(gauss0), 'o')
    plt.plot(ang_scales_u.arcmin, np.abs(fit_gauss_u), color='r')
    plt.plot(ang_scales_v.arcmin, np.abs(fit_gauss_v), color='r')
    plt.title('Amplitude vs Angular Scale')
    plt.ylabel('Amplitude (arbitrary)')
    plt.xlabel('Angular scale (arcmin)')
    plt.xscale('log')
    plt.savefig("visibility_fit_amp_angscale_{}.png".format(vis.time.isot[0]))

    # truths=[fit_amp_lm.params['I0'].value, sig_x.rad, sig_y.rad, rot_ang, 0]
    # fit_amp_vals = [fit_amp.params['I0'].value,
    #             fit_amp.params['sig_x'].value,
    #             fit_amp.params['sig_y'].value,
    #             fit_amp.params['theta'].value,
    #             fit_amp.params['C'].value
    #             ]
    # corner.corner(fit_amp.flatchain, labels=['I0', 'sig_x', 'sig_y', 'theta', 'C'])#, truths=truths)
    # plt.savefig("visibility_fit_corner_amp_{}.png".format(vis.time.isot[0]))
    #
    # fig, ax = plt.subplots(fit_amp.nvarys, sharex=True)
    # for i in range(fit_amp.nvarys):
    #     ax[i].plot(fit_amp.chain[:,:,i], 'k', alpha=0.3)
    #     # ax[i].hlines(truths[i], 0, fit_amp.chain.shape[0], colors='r', zorder=100)
    #     ax[i].hlines(fit_amp_vals[i], 0, fit_amp.chain.shape[0], colors='cyan', zorder=100)
    #     ax[i].set_ylabel(fit_amp.var_names[i])
    # ax[-1].set_xlabel("Step Number")
    # plt.savefig("visibility_fit_walkers_amp_{}.png".format(vis.time.isot[0]))
    #
    # truths=[x0.rad, y0.rad]
    # fit_vals = [fit_phase_mc.params['x0'].value, fit_phase_mc.params['y0'].value]
    # corner.corner(fit_phase_mc.flatchain, labels=['x0', 'y0'])#, truths=truths)
    # plt.savefig("visibility_fit_corner_phase_{}.png".format(vis.time.isot[0]))
    # fig, ax = plt.subplots(fit_phase_mc.nvarys, sharex=True)
    # for i in range(fit_phase_mc.nvarys):
    #     ax[i].plot(fit_phase_mc.chain[:,:,i], 'k', alpha=0.3)
    #     # ax[i].hlines(truths[i], 0, fit_phase_mc.chain.shape[0], colors='r', zorder=100)
    #     ax[i].hlines(fit_vals[i], 0, fit_phase_mc.chain.shape[0], colors='cyan', zorder=100)
    #     ax[i].set_ylabel(fit_phase_mc.var_names[i])
    # ax[-1].set_xlabel("Step Number")
    # plt.savefig("visibility_fit_walkers_phase_{}.png".format(vis.time.isot[0]))

    # truths=[fit_amp_lm.params['I0'].value, x0.rad, y0.rad, sig_x.rad, sig_y.rad, rot_ang, 0]
    fit_vals = [fit.params['I0'].value,
                fit.params['x0'].value,
                fit.params['y0'].value,
                fit.params['sig_x'].value,
                fit.params['sig_y'].value,
                #fit.params['theta'].value,
                fit.params['C'].value
                ]
    corner.corner(fit.flatchain, labels=['I0', 'x0', 'y0', 'sig_x', 'sig_y', 'C'])  # , truths=truths)
    plt.savefig("visibility_fit_corner_all_{}.png".format(vis.time.isot[0]))

    fig, ax = plt.subplots(fit.nvarys, sharex=True)
    for i in range(fit.nvarys):
        ax[i].plot(fit.chain[:, :, i], 'k', alpha=0.3)
        # ax[i].hlines(truths[i], 0, fit.chain.shape[0], colors='r', zorder=100)
        ax[i].hlines(fit_vals[i], 0, fit.chain.shape[0], colors='cyan', zorder=100)
        ax[i].set_ylabel(fit.var_names[i])
    ax[-1].set_xlabel("Step Number")
    plt.savefig("visibility_fit_walkers_all_{}.png".format(vis.time.isot[0]))
    print("Time to run {}".format(time.time() - t0))
    return fit, fit_amp, fit_phase


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

        fit, fit_amp, fit_phase = main(msin, trange)
        for f in [fit_amp, fit_phase, fit]:
            report_fit(f)
            print('\n')
        print("Aspect Ratio: {}".format(fit.params['sig_x'].value/fit.params['sig_y'].value))
        plt.show()
    else:
        df = pd.read_pickle(pickle)
        trange_list = []
        for i, t in enumerate(df[df.columns[0]]):
            tstart = Time(t)
            trange = TimeRange(tstart, tstart + 0.16 * u.s)
            trange_list.append(trange)
        with Pool() as pool:
            fit = pool.starmap(main, product([msin], trange_list))
        fit_df = pd.DataFrame([f.params.valuesdict() for f in fit])
        pickle_path = "burst_properties_visibility_fit{}.pkl".format(trange.start.isot[:10])
        fit_df.to_pickle(pickle_path)
