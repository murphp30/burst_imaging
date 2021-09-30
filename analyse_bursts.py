#!/usr/bin/env python
import glob
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.constants import eps0, e, m_e, R_sun
from astropy.coordinates import Angle
from astropy.time import Time
from matplotlib import dates
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit, minimize_scalar
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

def fit_line(x, m, c):
    return m*x + c

R_burst = minimize_scalar(find_burst_r)
R_burst = R_burst.x * u.m
R_sun_ang = Angle(R_sun_ang)
if not os.path.isfile('all_bursts_30MHz.pkl'):
    pickle_list = glob.glob('burst_properties_30MHz_2019*pkl')
    pickle_list.sort()
    df_list = []
    for pkl in pickle_list:
        df = pd.read_pickle(pkl)
        df = df.sort_index(axis=1)
        df_list.append(df)
    df = pd.concat(df_list, axis='columns')
    df.to_pickle('all_bursts_30MHz.pkl')
else:
    df = pd.read_pickle('all_bursts_30MHz.pkl')
# df = pd.read_pickle('uncal_all_bursts_30MHz.pkl')
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
df = df.T
bad_sig = 600/(2 * np.sqrt(2 * np.log(2))) #10arcmins
# df = df.where(df['amp'] > 2000)
df = df.where(df['sig_x'] != bad_sig)
df = df.dropna()
# best_times_file = "best_times.txt"
# best_times = np.loadtxt(best_times_file, dtype=str)
# df = df.loc[df.index.intersection(best_times)]
times = Time(list(df.index), format='isot')
xs = df['x0']
xs_m = (R_sun/R_sun_ang) * xs
ys = df['y0']
dx, dy, dfwhm_x, dfwhm_y = errors(df)
sig_x, sig_y = df['sig_x'].values, df['sig_y'].values
fwhm_x, fwhm_y = Angle(2 * np.sqrt(2 * np.log(2)) * sig_x * u.arcsec), Angle(
    2 * np.sqrt(2 * np.log(2)) * sig_y * u.arcsec)
sig_max = np.fmax(df['sig_x'].values, df['sig_y'].values)
sig_min = np.fmin(df['sig_x'].values, df['sig_y'].values)

fwhm_max = Angle(2 * np.sqrt(2 * np.log(2)) * sig_max * u.arcsec)
fwhm_min = Angle(2 * np.sqrt(2 * np.log(2)) * sig_min * u.arcsec)
fwhm_ratio = fwhm_x / fwhm_y
dfwhm_ratio = fwhm_ratio * np.sqrt((dfwhm_x / fwhm_x) ** 2 + (dfwhm_y / fwhm_y) ** 2)

best_fit, _ = curve_fit(fit_line, np.abs(xs_m/R_burst), fwhm_ratio, sigma=dfwhm_ratio)
# fig, ax = plt.subplots()
# ax.plot(times.plot_date, fwhm_max/R_sun_ang, 'o')
# ax.xaxis_date()
date_format = dates.DateFormatter("%Y-%m-%d")
# ax.xaxis.set_major_formatter(date_format)
# plt.xlabel('Time (UTC)')
# plt.ylabel(r'Major axis (R$_{\odot})$')
#
# fig, ax = plt.subplots()
# ax.plot(times.plot_date, fwhm_min/R_sun_ang, 'o')
# ax.xaxis_date()
# ax.xaxis.set_major_formatter(date_format)
# plt.xlabel('Time (UTC)')
# plt.ylabel(r'Minor axis (R$_{\odot})$')

# fig, ax = plt.subplots(figsize=(8, 7))
# ax.errorbar(times.plot_date, fwhm_ratio , dfwhm_ratio, marker='o', ls='')
# ax.xaxis_date()
# date_format = dates.DateFormatter("%m-%d")
# ax.xaxis.set_major_formatter(date_format)
# plt.xlabel('Date')
# plt.ylabel('FWHM ratio')
# plt.savefig('fwhm_ratio_date_modelfit.png')

# fig, ax = plt.subplots(figsize=(8, 7))
# ax.errorbar(times.plot_date, fwhm_y / R_sun_ang, dfwhm_y / R_sun_ang, marker='o', ls='')
# ax.xaxis_date()
# ax.xaxis.set_major_formatter(date_format)
# plt.xlabel('Date')
# plt.ylabel(r'FWHMy (R$_{\odot})$')
# plt.savefig('fwhmy_date_modelfit.png')
# #
# fig, ax = plt.subplots(figsize=(8,7))
# ax.errorbar(times.plot_date, fwhm_ratio, dfwhm_ratio,  marker='o', ls='')
# ax.xaxis_date()
# ax.xaxis.set_major_formatter(date_format)
# plt.xlabel('Date')
# plt.ylabel('FWHM ratio')
# plt.savefig('fwhmratio_date_bad.png')

# fig, ax = plt.subplots(figsize=(8,8))
# ax.errorbar(times.plot_date, xs, dx, marker='o', ls='')
# ax.xaxis_date()
# ax.xaxis.set_major_formatter(date_format)
# plt.xlabel('Date')
# plt.ylabel('X position (arcsec)')
# plt.savefig('x_date_bad.png')
#
# fig, ax = plt.subplots(figsize=(8,8))
# ax.errorbar(times.plot_date, ys, dy, marker='o', ls='')
# ax.xaxis_date()
# ax.xaxis.set_major_formatter(date_format)
# plt.xlabel('Date')
# plt.ylabel('Y position (arcsec)')
# plt.savefig('y_date_bad.png')

# fig, ax = plt.subplots(figsize=(8,8))
# ax.errorbar(xs, ys, dy, dx, ls='')
# sc = ax.scatter(xs, ys, c=times.plot_date)
# sun_circle = Circle((0,0), radius=R_sun_ang.arcsec, color='r', fill=False)
# plt.xlabel('X position (arcsec)')
# plt.ylabel('Y position (arcsec)')
# fig.colorbar(sc, ax=ax)
# ax.add_patch(sun_circle)
# sc.colorbar.ax.yaxis_date()
# sc.colorbar.ax.yaxis.set_major_formatter(date_format)
# ax.set_xlim(-2000,2000)
# ax.set_ylim(-2000,2000)
# plt.savefig('x_y_time_bad.png')

# fig, ax = plt.subplots(figsize=(11, 10))
# # ax.errorbar(xs, ys, dy, dx, ls='')
# sc = ax.scatter(xs, ys, c=fwhm_x / R_sun_ang, s=fwhm_x.arcsec)
# sun_circle = Circle((0, 0), radius=R_sun_ang.arcsec, color='r', fill=False)
# plt.xlabel('X position (arcsec)')
# plt.ylabel('Y position (arcsec)')
# fig.colorbar(sc, ax=ax)
# ax.add_patch(sun_circle)
# sc.colorbar.set_label(r'FWHM x(R$_{\odot})$')
# ax.set_xlim(-2000, 2000)
# ax.set_ylim(-2000, 2000)
# ax.set_aspect('equal')
# plt.savefig('x_y_fwhmx_bad.png')
# #
# fig, ax = plt.subplots(figsize=(11, 10))
# # ax.errorbar(xs, ys, dy, dx, ls='')
# sc = ax.scatter(xs, ys, c=fwhm_y / R_sun_ang, s=fwhm_y.arcsec)
# sun_circle = Circle((0, 0), radius=R_sun_ang.arcsec, color='r', fill=False)
# plt.xlabel('X position (arcsec)')
# plt.ylabel('Y position (arcsec)')
# fig.colorbar(sc, ax=ax)
# ax.add_patch(sun_circle)
# sc.colorbar.set_label(r'FWHM Y (R$_{\odot})$')
# ax.set_xlim(-2000, 2000)
# ax.set_ylim(-2000, 2000)
# ax.set_aspect('equal')
# plt.savefig('x_y_fwhmy_bad.png')
#
# fig, ax = plt.subplots(figsize=(8, 7))
# # ax.errorbar(xs, ys, dy, dx, ls='')
# sc = ax.scatter(xs, ys, c=fwhm_ratio, s=100 * fwhm_ratio)
# sun_circle = Circle((0, 0), radius=R_sun_ang.arcsec, color='r', fill=False)
# plt.xlabel('X position (arcsec)')
# plt.ylabel('Y position (arcsec)')
# fig.colorbar(sc, ax=ax)
# ax.add_patch(sun_circle)
# sc.colorbar.set_label('FWHM ratio')
# ax.set_xlim(-2000, 2000)
# ax.set_ylim(-2000, 2000)
# plt.savefig('x_y_fwhm_ratio_bad.png')

# fig, ax = plt.subplots(figsize=(8, 7))
# ax.errorbar(xs, fwhm_x / R_sun_ang, dfwhm_x / R_sun_ang, marker='o', ls='')
# plt.xlabel('x arcsec')
# plt.ylabel(r'FWHMx (R$_{\odot})$')
# plt.savefig('fwhmx_x_bad.png')
#
# fig, ax = plt.subplots(figsize=(8, 7))
# ax.errorbar(xs, fwhm_y / R_sun_ang, dfwhm_y / R_sun_ang, marker='o', ls='')
# plt.xlabel('x arcsec')
# plt.ylabel(r'FWHMy (R$_{\odot})$')
# plt.savefig('fwhmy_x_bad.png')
# plt.show()
fig, ax = plt.subplots(figsize=(8, 7))
ax.errorbar(xs, fwhm_ratio, dfwhm_ratio, marker='o', ls='')
plt.xlabel('x arcsec')
plt.ylabel('FWHM ratio')
plt.savefig('fwhm_ratio_x.png')

fig, ax = plt.subplots(figsize=(8, 7))
ax.errorbar(np.abs(xs_m/R_burst), fwhm_ratio, dfwhm_ratio, marker='o', ls='')
ax.plot(np.abs(xs_m/R_burst), fit_line(np.abs(xs_m/R_burst),*best_fit), 'r', zorder=100)
plt.xlim((0,1))
plt.xlabel(r'$\sin{\theta_s}$')
plt.ylabel('FWHM ratio')
plt.savefig('fwhm_ratio_sintheta.png')

# plt.savefig('fwhmy_x_bad.png')
fig, ax = plt.subplots(figsize=(11, 10))
# ax.errorbar(xs, ys, dy, dx, ls='')
sc = ellipses(df.x0, df.y0, fwhm_x, fwhm_y, df.theta, c=fwhm_ratio, alpha=0.5)
sun_circle = Circle((0, 0), radius=R_sun_ang.arcsec, color='r', fill=False)
plt.xlabel('X position (arcsec)')
plt.ylabel('Y position (arcsec)')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(sc, ax=ax)
ax.add_patch(sun_circle)
sc.colorbar.set_label('FWHM ratio')
ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_aspect('equal')
plt.savefig('burst_ellipses_modelfits.png')
