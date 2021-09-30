#!/usr/bin/env python
import glob
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.constants import eps0, e, m_e, R_sun
from astropy.coordinates import Angle, EarthLocation, SkyCoord, earth
from astropy.time import Time
from matplotlib import dates
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize_scalar
from sunpy.coordinates import frames, sun
from sunpy.sun.constants import average_angular_size as R_sun_ang


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

if not os.path.isfile('all_bursts_30MHz_good_times_serial_visfit.pkl'):
    pickle_list = glob.glob('burst_properties_30MHz_good_times_serial_visibility_fit_2019*.pkl')
    pickle_list.sort()
    df_list = []
    for pkl in pickle_list:
        df = pd.read_pickle(pkl)
        df = df.sort_index(axis=1)
        df_list.append(df)
    df = pd.concat(df_list)
    df.to_pickle('all_bursts_30MHz_good_times_serial_visfit.pkl')
else:
    df = pd.read_pickle('all_bursts_30MHz_good_times_serial_visfit.pkl')
# df = pd.read_pickle('uncal_all_bursts_30MHz_visfit.pkl')
# if not os.path.isfile('avg_bursts.pkl'):
#     pickle_list = glob.glob('burst_properties2019*pkl')
#     pickle_list.sort()
#     mean_df_list = []
#     for pkl in pickle_list:
#         mean_df = pd.read_pickle(pkl)
#         # df = df.sort_index(axis=1)
#         mean_df = mean_df.mean(axis=1)
#         mean_df_list.append(mean_df)
#     mean_df = pd.concat(mean_df_list, axis='columns')
#     mean_df.to_pickle('avg_bursts.pkl')
# else:
#     mean_df = pd.read_pickle('avg_bursts.pkl')
#     mean_df = mean_df.T
# df = df.T
bad_sig = Angle(15*u.arcmin).rad/(2 * np.sqrt(2 * np.log(2))) #20arcmins

# df = df.drorp(columns=['theta', 'theta_stderr'])
# df = df.where(df['sig_x'] > bad_sig)
# df = df.where(df['I0'] > 5e5)
# df = df.where(np.abs(df['redchi'] - 1) < 0.0005)
# df = df.dropna()

best_times_file = "good_enough_times.txt"#"best_times.txt"
best_times = np.loadtxt(best_times_file, dtype=str)
df = df.loc[df.index.intersection(best_times)]
times = Time(list(df.index), format='isot')
thetas = Angle(df['theta'] + sun.P(times).rad, u.rad).deg
xs = Angle(df['x0']*u.rad).arcsec
xs_m = (R_sun/R_sun_ang) * (xs*u.arcsec)
ys = Angle(df['y0']*u.rad).arcsec
# dx, dy, dfwhm_x, dfwhm_y = errors(df)
dx = Angle(df['x0_stderr']*u.rad).arcsec
dy = Angle(df['y0_stderr']*u.rad).arcsec
sig_x, sig_y = Angle(df['sig_x'].values*u.rad).arcsec, Angle(df['sig_y'].values*u.rad).arcsec
fwhm_x, fwhm_y = Angle(2 * np.sqrt(2 * np.log(2)) * sig_x * u.arcsec), Angle(
    2 * np.sqrt(2 * np.log(2)) * sig_y * u.arcsec)
dfwhm_x = 2 * np.sqrt(2 * np.log(2)) * Angle(df['sig_x_stderr'].values*u.rad).arcmin
dsig_y = np.sqrt(df['sig_x_stderr']**2 + df['delta_stderr']**2)
dfhwm_y = 2 * np.sqrt(2 * np.log(2)) * Angle(dsig_y*u.rad).arcmin
# sig_max = np.fmax(sig_x, sig_y)#(df['sig_x'].values, df['sig_y'].values)
# sig_min = np.fmin(sig_x, sig_y)#(df['sig_x'].values, df['sig_y'].values)
#
# fwhm_max = Angle(2 * np.sqrt(2 * np.log(2)) * sig_max * u.arcsec)
# fwhm_min = Angle(2 * np.sqrt(2 * np.log(2)) * sig_min * u.arcsec)
fwhm_ratio = sig_x/sig_y
dfwhm_ratio = fwhm_ratio*np.sqrt((df['sig_x_stderr']/df['sig_x'])**2 + (np.sqrt(df['sig_x_stderr']**2 + df['delta_stderr']**2)/df['sig_y'])**2)
# fwhm_ratio = fwhm_min / fwhm_max
# dfwhm_ratio = fwhm_ratio * np.sqrt((dfwhm_x / fwhm_x) ** 2 + (dfwhm_y / fwhm_y) ** 2)

core_ITRF = np.array((3826577.462, 461022.624, 5064892.526))
lofar_loc = EarthLocation.from_geocentric(*core_ITRF, u.m)
lofar_gcrs = SkyCoord(lofar_loc.get_gcrs(times))

xs = []
ys = []
for burst_centre in df['burst_centre_coord']:
    solar_x =  burst_centre.Tx.arcsec
    solar_y = burst_centre.Ty.arcsec
    xs.append(solar_x)
    ys.append(solar_y)
xs_m = (R_sun/R_sun_ang) * (xs*u.arcsec)
ys_m = (R_sun/R_sun_ang) * (ys*u.arcsec)#seems redundant to multiply by arcsec again but doesn't work otherwise
date_format = dates.DateFormatter("%Y-%m-%d")


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
plt.savefig('fwhm_ratio_best_times.png')

fig, ax = plt.subplots(figsize=(8, 7))
ax.errorbar(np.abs(ys_m/R_burst), fwhm_ratio, dfwhm_ratio, marker='o', ls='')
# plt.xlim((0,1))
plt.xlabel(r'$\sin{\theta_t}$')
plt.ylabel('FWHM ratio')

fig, ax = plt.subplots(figsize=(11, 10))
# ax.errorbar(xs, ys, dy, dx, ls='')
sc = ellipses(xs, ys, fwhm_y, fwhm_x,Angle(df['theta']*u.rad).deg, c=fwhm_ratio, alpha=0.5, vmin=0, vmax=2)
sun_circle = Circle((0, 0), radius=R_sun_ang.arcsec, color='r', fill=False)
plt.xlabel('X position (arcsec)')
plt.ylabel('Y position (arcsec)')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(sc, ax=ax)
ax.add_patch(sun_circle)
sc.colorbar.set_label('Aspect ratio')
ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_aspect('equal')
plt.savefig('burst_ellipses_best.png')
"""

params = [df['I0'], fwhm_x.arcmin, fwhm_y.arcmin, fwhm_ratio, Angle(df['theta'], u.rad).deg]
# params = [df['I0'], df['x0'], df['y0'], df['sig_x'], df['sig_y'], df['theta']]
param_names = ['Intensity (arbitrary)', 'FWHM X (arcmin)', 'FWHM Y (arcmin)', 'FWHM ratio', 'Position angle (degrees)']
# param_names = ['Intensity (arbitrary)', 'X (rad)', 'Y (rad)', 'sig X (rad)', 'sig Y (rad)','Position angle (rad)']
param_errors = [df['I0_stderr'], dfwhm_x, dfhwm_y, dfwhm_ratio, Angle(df['theta_stderr'], u.rad).deg]
# param_errors = [df['I0_stderr'], df['x0_stderr'], df['y0_stderr'], df['sig_x_stderr'], np.sqrt(df['sig_x_stderr']**2 +df['delta_stderr']**2), df['theta_stderr']]
for param, param_error, param_name in zip(params, param_errors, param_names):
    fig, ax = plt.subplots(figsize=(11, 10))
    # ax.errorbar(xs, param, param_error, dx, ls='', marker='o')
    # # ax.errorbar(df['x0'], param, param_error, df['x0_stderr'], ls='', marker='o')
    #
    # ax.set_xlabel('Solar X (arcsec)')
    # ax.set_ylabel(param_name)

    ax.hist(param, bins='auto')
    ax.set_xlabel(param_name)

plt.show()