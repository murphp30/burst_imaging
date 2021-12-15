#!/usr/bin/env python

import argparse
import glob
import os

import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.constants import eps0, e, m_e, R_sun
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.time import Time
from matplotlib import dates
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize_scalar
from sunpy.coordinates import frames, sun
from sunpy.sun.constants import average_angular_size as R_sun_ang

plt.rcParams.update({'font.size': 14})

def ellipses(x, y, w, h=None, rot=0.0, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of ellipses. From https://gist.github.com/syrte/592a062c562cd2a98a83
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of ellipses.
    w, h : scalar or array_like, shape (n, )
        Total length (diameter) of horizontal/vertical axis.
        `h` is set to be equal to `w` by default, ie. circle.
    rot : scalar or array_like, shape (n, )
        Rotation in degrees (anti-clockwise).
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    ellipses(a, a, w=4, h=a, rot=a*30, c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    if h is None:
        h = w

    zipped = np.broadcast(x, y, w, h, rot)
    patches = [Ellipse((x_, y_), w_, h_, rot_)
               for x_, y_, w_, h_, rot_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection

def errors(df):
    """
    Define errors of gaussian fit using Kontar et al. 2017 and Condon 1997
    https://arxiv.org/abs/1708.06505
    https://ui.adsabs.harvard.edu/abs/1997PASP..109..166C/abstract
    Here we assume error in flux (dF) is 1% of peak (S_0)
    angular resolution h is taken from major axis of clean beam = 465.67arcsec

    Input: df = `pandas.DataFrame` dataframe of measurements
    """
    dx = np.sqrt((2 / np.pi) * (df['sig_x'] / df['sig_y'])) * 0.01 * Angle(465.67 * u.arcsec)
    dy = np.sqrt((2 / np.pi) * (df['sig_y'] / df['sig_x'])) * 0.01 * Angle(465.67 * u.arcsec)
    sig_x, sig_y = df['sig_x'].values, df['sig_y'].values
    fwhm_x, fwhm_y = Angle(2 * np.sqrt(2 * np.log(2)) * sig_x * u.arcsec), \
                     Angle(2 * np.sqrt(2 * np.log(2)) * sig_y * u.arcsec)
    dfwhm_x = fwhm_x * 4 * np.sqrt(np.log(2)) * 0.01 * (Angle(465.67 * u.arcsec) / np.sqrt(np.pi * fwhm_x * fwhm_y))
    dfwhm_y = fwhm_y * 4 * np.sqrt(np.log(2)) * 0.01 * (Angle(465.67 * u.arcsec) / np.sqrt(np.pi * fwhm_x * fwhm_y))
    # dA = 2*np.pi*fwhm_x*fwhm_y*0.01*(Angle(465.67*u.arcsec)/np.sqrt(np.pi*fwhm_x*fwhm_y))
    return dx, dy, dfwhm_x, dfwhm_y

def density_from_plasma_freq(freq):
    """
    Compute plasma density from given frequency assuming it's the plasma frequency.
    Inputs: freq; plasma frequency, `astropy.units.quantity.Quantity`
    """
    omega_p = (2*np.pi)*freq
    N_e = omega_p**2 * (eps0.si*m_e.si)/(e.si**2)
    return N_e.decompose()

def density_3_pl(r):
    #Three power law density used by Kontar et al. 2019
    n = (4.8e9*((R_sun/r)**14)) +( 3e8*((R_sun/r)**6) )+ (1.4e6*((R_sun/r)**2.3))
    n = n*u.cm**(-3)
    return n.to(u.m**(-3))

def find_burst_r(r):
    freq = 30.46875*u.MHz
    n_p = density_from_plasma_freq(freq)
    return abs(n_p.value - density_3_pl(r * u.m).value)

R_burst = minimize_scalar(find_burst_r)
R_burst = R_burst.x * u.m
R_sun_ang = Angle(R_sun_ang)

parser = argparse.ArgumentParser()
parser.add_argument('--df_file', help='Input dataframe pickle.')
args = parser.parse_args()
df_file = args.df_file
if df_file is None:
    if not os.path.isfile('all_bursts_30MHz_fit_on_121021.pkl'): #('all_bursts_30MHz_good_times_visfit.pkl'):#
        pickle_list = glob.glob('burst_properties_30MHz_fit_on_121021_visibility_fit_2019*.pkl')#('burst_properties_30MHz_good_times_visibility_fit_2019*.pkl')#
        pickle_list.sort()
        df_list = []
        for pkl in pickle_list:
            df = pd.read_pickle(pkl)
            df = df.sort_index(axis=1)
            df_list.append(df)
        df = pd.concat(df_list)
        df.to_pickle('all_bursts_30MHz_fit_on_121021.pkl')#('all_bursts_30MHz_good_times_visfit.pkl')#
    else:
        df = pd.read_pickle('all_bursts_30MHz_fit_on_121021.pkl')#('burst_properties_30MHz_good_times_visibility_fit_2019-04-12.pkl')#('all_bursts_30MHz_good_times_visfit.pkl')#
else:
    df = pd.read_pickle(df_file)

if not os.path.isfile('all_burst_areas.pkl'):
    area_pickle_list = glob.glob('area_df_2019*.pkl')
    area_pickle_list.sort()
    area_df_list = []
    for pkl in area_pickle_list:
        area_df = pd.read_pickle(pkl)
        area_df = area_df.sort_index(axis=1)
        area_df_list.append(area_df)
    area_df = pd.concat(area_df_list)
    area_df.to_pickle('all_burst_areas.pkl')
else:
    area_df = pd.read_pickle('all_burst_areas.pkl')

bad_sig = Angle(15*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))) #20arcmins
best_times_file = "best_times.txt" #"good_enough_times.txt"#
best_times = np.loadtxt(best_times_file, dtype=str)
df = df.loc[df.index.intersection(best_times)]
df = df[~df.index.duplicated(keep='first')] #weird duplicated index
times = Time(list(df.index), format='isot')
area_times = Time([*area_df.index])


#have to do this due to an off by one error somewhere
area_best_times = []
df_best_times = []
for t in Time(best_times):
    area_best_times.append(np.abs(t - area_times).argmin())
    df_best_times.append(np.abs(t - times).argmin())

# df = df.loc[df.index[df_best_times]]
# area_df = area_df.loc[area_df.index[area_best_times]]

times = Time(list(df.index), format='isot')
area_times = Time([*area_df.index])
thetas = Angle(df['theta'] - sun.P(times).rad, u.rad).deg

area_growth_rates = []
area_growth_errors = []
for fit in area_df['area_growth_fit']:
    area_growth_rates.append(fit.params['slope'].value)
    area_growth_errors.append(fit.params['slope'].stderr)
dx = Angle(df['x0_stderr']*u.rad).arcsec
dy = Angle(df['y0_stderr']*u.rad).arcsec
sig_x, sig_y = Angle(df['sig_x'].values*u.rad).arcsec, Angle(df['sig_y'].values*u.rad).arcsec
fwhm_x, fwhm_y = Angle(2 * np.sqrt(2 * np.log(2)) * sig_x * u.arcsec), Angle(
    2 * np.sqrt(2 * np.log(2)) * sig_y * u.arcsec)
dfwhm_x = 2 * np.sqrt(2 * np.log(2)) * Angle(df['sig_x_stderr'].values*u.rad).arcmin
dsig_y = np.sqrt(df['sig_x_stderr']**2 + df['delta_stderr']**2)
dfwhm_y = 2 * np.sqrt(2 * np.log(2)) * Angle(dsig_y*u.rad).arcmin

fwhm_ratio = sig_x/sig_y
dfwhm_ratio = fwhm_ratio*np.sqrt((df['sig_x_stderr']/df['sig_x'])**2 + (np.sqrt(df['sig_x_stderr']**2 + df['delta_stderr']**2)/df['sig_y'])**2)

core_ITRF = np.array((3826577.462, 461022.624, 5064892.526))
lofar_loc = EarthLocation.from_geocentric(*core_ITRF, u.m)
lofar_gcrs = SkyCoord(lofar_loc.get_gcrs(times))

xs = []
ys = []
for burst_centre in df['burst_centre_coord']:
    solar_x = burst_centre.Tx.arcsec
    solar_y = burst_centre.Ty.arcsec
    xs.append(solar_x)
    ys.append(solar_y)

lats = []
lons = []
for burst_centre in df['burst_centre_coord']:
    car_lat = burst_centre.transform_to(frames.HeliographicCarrington).lat.deg
    car_lon = burst_centre.transform_to(frames.HeliographicCarrington).lon.deg
    lats.append(car_lat)
    lons.append(car_lon)

xs_m = (R_sun/R_sun_ang) * (xs*u.arcsec)
ys_m = (R_sun/R_sun_ang) * (ys*u.arcsec)#seems redundant to multiply by arcsec again but doesn't work otherwise
date_format = dates.DateFormatter("%Y-%m-%d")

# x_asec_arr = []
# for x in xs:
#     x_asec_arr.append(np.linspace(x,0))
slopes = np.array(ys)/np.array(xs)
# lines = slopes[:, np.newaxis] * x_asec_arr
#
burst_slopes = np.tan(Angle(thetas*u.deg).rad)
#
# burst_min_axes = []
# for i in range(len(xs)):
#     burst_min_axes.append(np.linspace(xs[i],xs[i]+0.5*fwhm_x.arcsec[i]))
#
# burst_lines = burst_slopes[:,np.newaxis] * (np.array(burst_min_axes) - np.array(xs)[:,np.newaxis]) + np.array(ys)[:,np.newaxis]
relative_angles = Angle(np.arctan((burst_slopes - slopes)/(1 + (burst_slopes*slopes)))*u.rad).deg

# burst_slopes_a = np.tan(df['theta'])
# relative_angles_a = Angle(np.arctan((burst_slopes_a - slopes)/(1 + (burst_slopes_a*slopes)))*u.rad).deg
# mags, bmags = [], []
# for i in range(len(lines)):
#     x0, y0 = x_asec_arr[i][0], lines[i][0]
#     x1, y1 = x_asec_arr[i][-1], lines[i][-1]
#     mag = np.sqrt((x1-x0)**2 + (y1-y0)**2)
#     mags.append(mag)
#     bx1, by1 = x_asec_arr[i][-1], burst_lines[i][-1]
#     bmag = np.sqrt((bx1-x0)**2 + (by1-y0)**2)
#     bmags.append(bmag)
# mags = np.array(mags)
# bmags = np.array(bmags)
# dots = []
# for i in range(len(lines)):
#     dots.append(np.dot(lines[i], burst_lines[i]))


# fig, ax = plt.subplots(figsize=(8, 7))
# ax.errorbar(np.abs(xs_m/R_burst),fwhm_x.arcsec/R_sun_ang.arcsec, dfwhm_x.arcsec/R_sun_ang.arcsec, marker='o', ls='')
# # plt.xlim((0,1))
# plt.xlabel(r'$\sin{\theta_s}$')
# plt.ylabel('FWHM x')
#
# fig, ax = plt.subplots(figsize=(8, 7))
# ax.errorbar(np.abs(xs_m/R_burst), fwhm_y.arcsec/R_sun_ang.arcsec, dfwhm_y.arcsec/R_sun_ang.arcsec, marker='o', ls='')
# # plt.xlim((0,1))
# plt.xlabel(r'$\sin{\theta_s}$')
# plt.ylabel('FWHM y')
"""
fig, ax = plt.subplots(figsize=(8, 7))
ax.errorbar(xs, fwhm_ratio, dfwhm_ratio, dx, marker='o', ls='')
# plt.xlim((0,1))
plt.xlabel('Solar X arcsec')
plt.ylabel('FWHM ratio')

fig, ax = plt.subplots(figsize=(8, 7))
ax.errorbar(ys, fwhm_ratio, dfwhm_ratio, dy, marker='o', ls='')
# plt.xlim((0,1))
plt.xlabel('Solar Y arcsec')
plt.ylabel('FWHM ratio')

fig, ax = plt.subplots(figsize=(8, 7))
ax.errorbar(np.abs(xs_m/R_burst), fwhm_ratio, dfwhm_ratio, marker='', ls='')
scatter = ax.scatter(np.abs(xs_m/R_burst), fwhm_ratio,c=np.abs(ys), zorder=100)
# plt.xlim((0,1))
fig.colorbar(scatter, ax=ax)
scatter.colorbar.set_label('distance from equator (arcsec)')
plt.xlabel(r'$\sin{\theta_s}$')
plt.ylabel('FWHM ratio')
# plt.savefig('fwhm_ratio_best_times.png')

fig, ax = plt.subplots(figsize=(8, 7))
ax.errorbar(np.abs(ys_m/R_burst), fwhm_ratio, dfwhm_ratio, marker='o', ls='')
# plt.xlim((0,1))
plt.xlabel(r'$\sin{\theta_t}$')
plt.ylabel('FWHM ratio')
"""
fig, ax = plt.subplots(figsize=(11, 10))
sc = ellipses(xs, ys, fwhm_x, fwhm_y, thetas, c=fwhm_ratio, alpha=0.5)
sun_circle = Circle((0, 0), radius=R_sun_ang.arcsec, color='r', fill=False)
plt.xlabel('X position (arcsec)')
plt.ylabel('Y position (arcsec)')
cb = fig.colorbar(sc, ax=ax)
# loc = dates.AutoDateLocator()
# cb.ax.yaxis.set_major_locator(loc)
# cb.ax.yaxis.set_major_formatter(dates.ConciseDateFormatter(loc))
ax.add_patch(sun_circle)
sc.colorbar.set_label('Aspect Ratio')
# cb.ax.yaxis_date()
# date_format = dates.DateFormatter("%H:%M:%S")
# cb.ax.yaxis.set_major_formatter(date_format)
ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_aspect('equal')
# plt.savefig('burst_ellipses_best.png')

# fig, ax = plt.subplots(figsize=(11, 10))
# ellipses(lons, lats, fwhm_x.arcmin, fwhm_y.arcmin, thetas, alpha=0.5)
# ax.set_xlim(0, 360)
# # ax.set_ylim(-90, 90)
# ax.set_xlabel('Carrington Longitude (deg)')
# ax.set_ylabel('Carrington Latitude (deg)')
# # plt.savefig('burst_ellipses_carrington.png')
params = [df['I0'], xs, ys, fwhm_x.arcmin, fwhm_y.arcmin, fwhm_ratio, Angle(df['theta'], u.rad).deg]
# # params = [df['I0'], df['x0'], df['y0'], df['sig_x'], df['sig_y'], df['theta']]
param_names = ['Intensity (arbitrary)', 'Solar X (arcsec)', 'Solar Y (arcsec)', 'FWHM X (arcmin)', 'FWHM Y (arcmin)', 'FWHM ratio', 'Position angle (degrees)']
# # param_names = ['Intensity (arbitrary)', 'X (rad)', 'Y (rad)', 'sig X (rad)', 'sig Y (rad)','Position angle (rad)']
param_errors = [df['I0_stderr'], dx, dy, dfwhm_x, dfwhm_y, dfwhm_ratio, Angle(df['theta_stderr'], u.rad).deg]

fwhms = [fwhm_x.arcmin, fwhm_y.arcmin, fwhm_ratio]
fwhm_names = ['FWHM X (arcmin)', 'FWHM Y (arcmin)', 'FWHM ratio']
fwhm_errors = [dfwhm_x, dfwhm_y, dfwhm_ratio]

# for param, param_error, param_name in zip(fwhms, fwhm_errors, fwhm_names):
#     fig, ax = plt.subplots(1,2, sharey=True, figsize=(14, 7))
#     ax[0].errorbar(np.abs(xs), param, param_error, dx, ls='', marker='o')
#     ax[1].errorbar(np.abs(ys), param, param_error, dy, ls='', marker='o')
#     # ax.errorbar(df['x0'], param, param_error, df['x0_stderr'], ls='', marker='o')
#
#     ax[0].set_xlabel('Absolute Solar X (arcsec)')
#     ax[1].set_xlabel('Absolute Solar Y (arcsec)')
#     ax[0].set_ylabel(param_name)


# param_errors = [df['I0_stderr'], df['x0_stderr'], df['y0_stderr'], df['sig_x_stderr'], np.sqrt(df['sig_x_stderr']**2 +df['delta_stderr']**2), df['theta_stderr']]
# for param, param_error, param_name in zip(params, param_errors, param_names):
#     fig, ax = plt.subplots(figsize=(9, 7))
#     ax.errorbar(param, area_growth_rates, area_growth_errors, param_error, ls='', marker='o')
#     ax.set_xlabel(param_name)
#     ax.set_ylabel('Area growth rate (arcmin/s)')
#     # ax.errorbar(xs, param, param_error, dx, ls='', marker='o')
#     # # ax.errorbar(df['x0'], param, param_error, df['x0_stderr'], ls='', marker='o')
#     #
#     # ax.set_xlabel('Solar X (arcsec)')
#     # ax.set_ylabel(param_name)
#
#     ax.hist(param, bins='auto')
#     ax.set_xlabel(param_name)

# fig, ax = plt.subplots(len(params), len(params), figsize=(10, 10))
# for i in range(len(params)):
#     for j in range(len(params)):
#         fig, ax = plt.subplots(figsize=(9,7))
#         ax.errorbar(params[i], params[j], param_errors[j], param_errors[i], ls='', marker='o')
#         ax.set_ylabel(param_names[j])
#         ax.set_xlabel(param_names[i])
# #         plt.savefig("vis_fits_stats_{}_vs_{}.png".format(param_names[i], param_names[j]))
#         plt.close()

"""horrible ugly thing because sharing some axes in matplotlib is stupid"""
plt.figure(figsize=(14,10))
ax1 = plt.subplot(2,2,1)
ax1.errorbar(np.abs(xs), fwhms[0], fwhm_errors[0], dx, ls='', marker='o', label='FWHM X')
ax1.errorbar(np.abs(xs), fwhms[1], fwhm_errors[1], dx, ls='', marker='o', label='FWHM Y')
ax1.set_ylim(7,22)
ax1.annotate('a)', (0.05, 0.95), xycoords='axes fraction' )
ax1.legend()

plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_xticklines(), visible=False)

ax2 = plt.subplot(2,2,2, sharey=ax1)
ax2.errorbar(np.abs(ys), fwhms[0], fwhm_errors[0], dy, ls='', marker='o', label='FWHM X')
ax2.errorbar(np.abs(ys), fwhms[1], fwhm_errors[1], dy, ls='', marker='o', label='FWHM Y')
ax2.legend()
ax2.annotate('b)', (0.05, 0.95), xycoords='axes fraction' )
plt.setp(ax2.get_xticklabels(), visible=False)
# plt.setp(ax2.get_xticklines(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
# plt.setp(ax2.get_yticklines(), visible=False)

ax3 = plt.subplot(2,2,3, sharex=ax1)
ax3.errorbar(np.abs(xs), fwhms[2], fwhm_errors[2], dx, ls='', marker='o')
ax3.annotate('c)', (0.05, 0.95), xycoords='axes fraction' )
ax4 = plt.subplot(2,2,4, sharex=ax2, sharey=ax3)
ax4.errorbar(np.abs(ys), fwhms[2], fwhm_errors[2], dy, ls='', marker='o')
ax4.annotate('d)', (0.05, 0.95), xycoords='axes fraction' )
plt.setp(ax4.get_yticklabels(), visible=False)
# plt.setp(ax4.get_yticklines(), visible=False)

ax1.set_ylabel('FWHM (arcmin)')
ax3.set_xlabel('Absolute Solar X (arcsec)')
ax3.set_ylabel('Aspect ratio')
ax4.set_xlabel('Absolute Solar Y (arcsec)')
plt.tight_layout()
plt.savefig('fwhm_comparison.png')

hours = (times-times[0]).sec/60/60
plt.figure(figsize=(9,7))
plt.errorbar(hours, fwhms[0], fwhm_errors[0], ls='', marker='o', label='FWHM X')
plt.errorbar(hours, fwhms[1], fwhm_errors[1], ls='', marker='o', label='FWHM Y')
plt.legend()
plt.xlabel('Hours from observation start')
plt.ylabel('FWHM')
plt.savefig('fwhm_time_comp.png')

bins = np.arange(6, 21, step=1)
n_x, bins_x = np.histogram(fwhm_x.arcmin, bins=bins)
n_y, bins_y = np.histogram(fwhm_y.arcmin, bins=bins)
mean_x, mean_y = np.mean(fwhm_x.arcmin), np.mean(fwhm_y.arcmin)

fig, ax = plt.subplots(1,3, sharey=True, figsize=(18,6))
ax[0].hist(fwhm_x.arcmin, bins = bins_x, alpha=0.5, label="FWHMx")
ax[0].hist(fwhm_y.arcmin, bins = bins_y, alpha=0.5, label="FWHMy")
ax[0].vlines(bins_x[:-1], 0, n_x, colors='b', alpha=0.25)
ax[0].vlines(bins_y[:-1], 0, n_y, colors='r', alpha=0.25)

ax[0].axvline(mean_x, c='b', ls='--')
ax[0].axvline(mean_y, c='r', ls='--')
ax[0].set_xlabel('FWHM (arcmin)')
ax[0].set_ylabel('Number of bursts')
ax[0].legend()
ax[0].annotate('a)', (0.05, 0.95), xycoords='axes fraction' )
# plt.savefig('burst_fwhm_histogram.png')

mean_r = np.mean(fwhm_ratio)
n_r, bins_r = np.histogram(fwhm_ratio)
# fig, ax = plt.subplots(figsize=(9,7))
ax[1].hist(fwhm_ratio, bins = bins_r, alpha=0.5, color='purple')
ax[1].vlines(bins_r[:-1], 0, n_r, colors='purple', alpha=0.25)
ax[1].axvline(mean_r, c='purple', ls='--')
ax[1].set_xlabel('Aspect Ratio')
ax[1].annotate('b)', (0.05, 0.95), xycoords='axes fraction' )
# ax[1].set_ylabel('Number of bursts')
# plt.savefig('burst_fwhm_ratio_histogram.png')
# plt.close('all')
print("Mean FWMHx: {} arcmin \n Mean FWHMy: {} arcmin \n Mean Aspect Ratio: {}".format(*np.round((mean_x, mean_y, mean_r), 2)))

mean_rel = np.mean(relative_angles)
n_rel, bins_rel = np.histogram(relative_angles)
# fig, ax = plt.subplots(figsize=(9,7))
ax[2].hist(relative_angles, bins = bins_rel, alpha=0.5, color='green')
ax[2].vlines(bins_rel[:-1], 0, n_rel, colors='green', alpha=0.25)
ax[2].axvline(mean_rel, c='green', ls='--')
ax[2].set_xlabel('Relative Angle (degrees)')
ax[2].annotate('c)', (0.05, 0.95), xycoords='axes fraction' )
# ax[2].set_ylabel('Number of bursts')
# plt.savefig('burst_relative_angle_histogram.png')
plt.tight_layout()
plt.savefig('burst_3histogram.png')
time_from_start = np.arange(0,3, 0.16)

# fig, ax = plt.subplots()
# for i in range(len(area_df['areas_amin'])):
#     fig, ax = plt.subplots()
#     ax.plot(time_from_start[:-1], area_df['areas_amin'][i], 'o')
#     ax.plot(time_from_start[:-1], area_df['area_growth_fit'][i].best_fit)
#     fig.canvas.draw()
#     plt.pause(0.5)
#     plt.close()


plt.show()
