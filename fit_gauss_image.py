#!/usr/bin/env python

"""
Fit a gauss to a single burst in image space
Pearse Murphy 30/03/20 COVID-19
Takes fits file created by WSClean as input
"""

import argparse
import os
import warnings

import astropy.units as u
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy.map

from astropy.coordinates import Angle, SkyCoord
from lmfit import Parameters, Model
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from sunpy.coordinates.frames import Helioprojective
from sunpy.sun.constants import average_angular_size as R_av_ang

from icrs_to_helio import icrs_to_helio
from manual_clean import convolve_model

warnings.filterwarnings("ignore")


def rotate_coords(x, y, theta):
    # rotate coordinates x y anticlockwise by theta
    x_p = x * np.cos(theta) + y * np.sin(theta)
    y_p = -x * np.sin(theta) + y * np.cos(theta)
    return x_p, y_p

def gauss_2d(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
    """
    Create a flattened 2D gaussian.

    Parameters
    ----------
    xy : tuple, list
        A tuple of x and y coordinates.
    amp : float
        Amplitude of gaussian.
    x0 : float
        x coordinate of gaussian maximum.
    y0 : float
        y coordinate of gaussian maximum.
    sig_x : float
        Standard deviation in x direction.
    sig_y : float
        Standard deviation in y direction.
    theta : float
        Position angle of gaussian.
    offset : float
        Background level of gaussian.
    """
    #   x = xy.Tx.arcsec
    #   y = xy.Ty.arcsec
    (x, y) = xy
    x0 = float(x0)
    y0 = float(y0)
    x, y = rotate_coords(x, y, theta)
    x0, y0 = rotate_coords(x0, y0 , theta)
    # a = ((np.cos(theta) ** 2) / (2 * sig_x ** 2)) + ((np.sin(theta) ** 2) / (2 * sig_y ** 2))
    # b = ((np.sin(2 * theta)) / (4 * sig_x ** 2)) - ((np.sin(2 * theta)) / (4 * sig_y ** 2))
    # c = ((np.sin(theta) ** 2) / (2 * sig_x ** 2)) + ((np.cos(theta) ** 2) / (2 * sig_y ** 2))
    # g = amp * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))) + offset
    g = amp * np.exp(-(((x - x0) ** 2) / (2 * sig_x ** 2) + ((y - y0) ** 2) / (2 * sig_y ** 2))) + offset
    return g.ravel()


def log_likelihood(params, xy, data, err):
    """Calculate log of the likelihood function."""
    # amp, x0, y0, sig_x, sig_y, theta, offset, log_f = params
    amp, x0, y0, sig_x, sig_y, theta, offset = params

    model = gauss_2d(xy, amp, x0, y0, sig_x, sig_y, theta, offset)
    sigma2 = err ** 2  # + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((data - model) ** 2 / sigma2 + np.log(sigma2))


def log_prior(params, data, initial_params):
    """Calculate log of prior function"""
    # amp, x0, y0, sig_x, sig_y, theta, offset, log_f = params
    amp, x0, y0, sig_x, sig_y, theta, offset = params
    amp0, x00, y00, sig_x0, sig_y0, theta0, offset0 = initial_params

    if 0.75 * np.max(data) < amp < 1.1 * np.max(
            data) and x00 - 500.0 < x0 < x00 + 500.0 and y00 - 500.0 < y0 < y00 + 500.0 and \
            0.0 < sig_x < 2 * sig_x0 and 0.0 < sig_y < 2 * sig_y0 and 0.0 < theta < np.pi / 2 and \
            0 < offset < 0.25 * np.max(data):  # and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf


def log_probability(params, xy, data, err, initial_params):
    """Calculate log of posterior probabilty function"""
    lp = log_prior(params, data, initial_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, xy, data, err)


def make_init_params(smap, fwhm_x, fwhm_y, theta, offset):
    """
    Make dictionary of initial parameters to model gauss_2d

    Parameters
    ----------
    smap : `sunpy.map.mapbase.GenericMap`
        sunpy map to fit
    fwhm_x : float
        initial guess of the full width at half maximum height (FWHM) in the x direction in arcmin
    fwhm_y : float
        initial guess of the FWHM in the y direction in arcmin
    theta : float
        initial guess of position angle of gaussian
    offset : float
        initial guess of background level of gaussian
    """
    max_xy = np.where(smap.data == smap.data.max())
    max_pos = smap.pixel_to_world(max_xy[1][0] * u.pix, max_xy[0][0] * u.pix)
    # x and y positions are in the opposite places where you'd expect them and I don't
    # know why. This works so go with it.

    init_params = {"amp": smap.data.max(),
                   "x0": max_pos.Tx.arcsec,
                   "y0": max_pos.Ty.arcsec,
                   "sig_x": Angle(fwhm_x * u.arcmin).arcsec / (2 * np.sqrt(2 * np.log(2))),
                   "sig_y": Angle(fwhm_y * u.arcmin).arcsec / (2 * np.sqrt(2 * np.log(2))),
                   "theta": theta,
                   "offset": offset}
    return init_params


def make_params(smap, fwhm_x=10, fwhm_y=18, theta=0.1, offset=0):
    """
    create lmfit.Parameters object and set how much each can vary in model fitting

    Parameters
    ----------
    See `make_init_params`
    """
    init_params = make_init_params(smap, fwhm_x, fwhm_y, theta, offset)
    params = Parameters()
    params.add_many(("amp", init_params["amp"], True, 0.5 * init_params["amp"], None),
                    ("x0", init_params["x0"], True, init_params["x0"] - 150, init_params["x0"] + 150),
                    ("y0", init_params["y0"], True, init_params["y0"] - 150, init_params["y0"] + 150),
                    ("sig_x", init_params["sig_x"], True, 0, 2 * R_av_ang.value / (2 * np.sqrt(2 * np.log(2)))),
                    ("sig_y", init_params["sig_y"], True, 0, 2 * R_av_ang.value / (2 * np.sqrt(2 * np.log(2)))),
                    ("theta", init_params["theta"], False, -np.pi/6, np.pi/6),
                    ("offset", init_params["offset"], True, 0.01 * smap.data.min(), smap.data.min()))
    return params


def rotate_zoom(smap, x0, y0, theta):
    """Rotate and resize input map to be centred on the gaussian"""
    # shift = smap.shift(x0, y0)
    top_right = SkyCoord(x0 + 2000 * u.arcsec, y0 + 2000 * u.arcsec, frame=smap.coordinate_frame)
    bottom_left = SkyCoord(x0 - 2000 * u.arcsec, y0 - 2000 * u.arcsec, frame=smap.coordinate_frame)
    zoom = smap.submap(bottom_left, top_right=top_right)
    rot = zoom.rotate(-theta)
    return rot

def burst_centre(smap, gauss_centre):
    zoom_centre = smap.world_to_pixel(SkyCoord(gauss_centre))
    zoom_xy = sunpy.map.all_coordinates_from_map(smap)
    x_cen = int(zoom_centre.x.round().value)
    y_cen = int(zoom_centre.y.round().value)

    return x_cen, y_cen, zoom_xy
def plot_nice(smap, fit, save=True):
    """Plot rotated sunpy map and overlay gaussian fit"""
    smap.plot_settings['cmap'] = 'viridis'
    x0 = fit.best_values['x0'] * u.arcsec
    y0 = fit.best_values['y0'] * u.arcsec
    theta = Angle(fit.best_values['theta'] * u.rad)
    gauss_centre = SkyCoord(x0, y0, frame='helioprojective', observer=smap.observer_coordinate, obstime=smap.date)
    fit_map = sunpy.map.Map(fit.best_fit.reshape(smap.data.shape), smap.meta)
    rot_fit = rotate_zoom(fit_map, x0, y0, theta)
    rot_helio = rotate_zoom(smap, x0, y0, theta)

    # zoom_centre = rot_helio.world_to_pixel(SkyCoord(gauss_centre))
    # zoom_xy = sunpy.map.all_coordinates_from_map(rot_helio)
    # x_cen = int(zoom_centre.x.round().value)
    # y_cen = int(zoom_centre.y.round().value)
    x_cen, y_cen, zoom_xy = burst_centre(rot_helio, gauss_centre)

    new_dims = [99, 99] * u.pix  # Resample data to 100 * 100 for histogram plot
    helio_resample = rot_helio.resample(new_dims)
    # resample_xy = sunpy.map.all_coordinates_from_map(helio_resample)
    resample_x_cen, resample_y_cen, resample_xy = burst_centre(helio_resample, gauss_centre)

    # take a slice through the middle index.
    x_1D_helio, y_1D_helio = helio_resample.data[resample_y_cen, :], helio_resample.data[:, resample_x_cen]
    x_1D_fit, y_1D_fit = rot_fit.data[y_cen, :], rot_fit.data[:, x_cen]
    zoom_xarr = zoom_xy[y_cen, :]
    zoom_yarr = zoom_xy[:, x_cen]
    resample_xarr = resample_xy[resample_y_cen, :]
    resample_yarr = resample_xy[:, resample_x_cen]
    coord_x = rot_helio.pixel_to_world([0, (zoom_xy.shape[1] - 1)] * u.pix, [y_cen, y_cen] * u.pix)
    coord_y = rot_helio.pixel_to_world([x_cen, x_cen] * u.pix, [0, (zoom_xy.shape[0] - 1)] * u.pix)

    print(fit.fit_report())
    fwhmx = Angle((2 * np.sqrt(2 * np.log(2)) * fit.best_values['sig_x']) * u.arcsec).arcmin
    fwhmy = Angle((2 * np.sqrt(2 * np.log(2)) * fit.best_values['sig_y']) * u.arcsec).arcmin
    print('FWHMX: {} arcmin\nFWHMY: {} arcmin'.format(np.round(fwhmx, 2), np.round(fwhmy, 2)))

    hwhmx_pixels = fwhmx * 30 * u.arcsec / rot_helio.scale.axis1
    hwhmy_pixels = fwhmy * 30 * u.arcsec / rot_helio.scale.axis2
    coord_x_hwhml = rot_helio.pixel_to_world([x_cen, (zoom_xy.shape[1] - 1)] * u.pix,
                                             [y_cen - hwhmy_pixels.value, y_cen - hwhmy_pixels.value] * u.pix)
    coord_x_hwhmr = rot_helio.pixel_to_world([x_cen, (zoom_xy.shape[1] - 1)] * u.pix,
                                             [y_cen + hwhmy_pixels.value, y_cen + hwhmy_pixels.value] * u.pix)
    coord_y_hwhml = rot_helio.pixel_to_world([x_cen - hwhmx_pixels.value, x_cen - hwhmx_pixels.value, ] * u.pix,
                                             [y_cen, (zoom_xy.shape[0] - 1)] * u.pix)
    coord_y_hwhmr = rot_helio.pixel_to_world([x_cen + hwhmx_pixels.value, x_cen + hwhmx_pixels.value, ] * u.pix,
                                             [y_cen, (zoom_xy.shape[0] - 1)] * u.pix)
    beam_cen = [(x_cen - 200), (y_cen - 200)]
    # pdb.set_trace()
    # Plotting stuff
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4)
    ax = fig.add_subplot(gs[1:4, 0:3], projection=rot_helio)
    ax0 = fig.add_subplot(gs[0:1, 0:3])
    ax1 = fig.add_subplot(gs[1:4, 3:])
    ax_lg = fig.add_subplot(gs[0:1, 3])
    ax_lg.axis('off')
    helio_plot = rot_helio.plot(axes=ax, title='')
    rot_helio.draw_limb(ax)
    BMAJ, BMIN, BPA = [Angle(smap.meta[key], 'deg') for key in ['bmaj', 'bmin', 'bpa']]
    solar_PA = sunpy.coordinates.sun.P(smap.date).deg
    # patch is all in pixels. There's probably an easy way to get to WCS.
    beam = Ellipse((beam_cen[0], beam_cen[1]),
                   (BMAJ / abs(smap.scale.axis1)).value, (BMIN / abs(smap.scale.axis2)).value,
                   90 - BPA.deg + solar_PA + theta.deg,
                   color='w', ls='--', fill=False)
    ax.add_patch(beam)
    gr = rot_helio.draw_grid(ax)
    level = 100 * (0.5 * (fit.best_values['offset'] + np.max(fit_map.data))) / np.max(fit_map.data)
    rot_fit.draw_contours(axes=ax, levels=[level] * u.percent, colors=['red'])
    lon = helio_plot.axes.coords[0]
    lat = helio_plot.axes.coords[1]
    ax.plot_coord(coord_x, '--', color='white')
    ax.plot_coord(coord_y, '--', color='white')
    ax.plot_coord(coord_x_hwhml, '-', color='grey')
    ax.plot_coord(coord_x_hwhmr, '-', color='grey')
    ax.plot_coord(coord_y_hwhml, '-', color='grey')
    ax.plot_coord(coord_y_hwhmr, '-', color='grey')

    # top plot
    ax0.plot(resample_xarr.Tx.arcmin, x_1D_helio, drawstyle='steps-mid', label="LOFAR source")
    ax0.plot(zoom_xarr.Tx.arcmin, x_1D_fit, label="Gaussian fit")
    ax0.axvline(coord_y_hwhml[0].Tx.arcmin, color='grey')
    ax0.axvline(coord_y_hwhmr[0].Tx.arcmin, color='grey')
    ax0.annotate("",
                 xy=(coord_y_hwhml[0].Tx.arcmin, 0.5 * (fit.best_values['offset'] + np.max(x_1D_fit))),
                 xycoords="data",
                 xytext=(coord_y_hwhmr[0].Tx.arcmin, 0.5 * np.max(x_1D_fit)),
                 textcoords="data",
                 arrowprops=dict(arrowstyle="<->"))
    ax0.text(0.5, 0.4, "{:.2f}'".format(fwhmx),
             horizontalalignment="center", transform=ax0.transAxes)
    ax0.autoscale(axis="x", tight=True)
    ax0.set_ylabel("Intensity (relative)")

    # right plot
    ax1.plot(y_1D_helio, resample_yarr.Ty.arcmin, drawstyle='steps-mid')
    ax1.plot(y_1D_fit, zoom_yarr.Ty.arcmin)
    ax1.axhline(coord_x_hwhml[0].Ty.arcmin, color='grey')
    ax1.axhline(coord_x_hwhmr[0].Ty.arcmin, color='grey')
    ax1.annotate("",
                 xy=(0.5 * (fit.best_values['offset'] + np.max(y_1D_fit)), coord_x_hwhml[0].Ty.arcmin),
                 xycoords="data",
                 xytext=(0.5 * np.max(y_1D_fit), coord_x_hwhmr[0].Ty.arcmin),
                 textcoords="data",
                 arrowprops=dict(arrowstyle="<->"))
    ax1.text(0.5, 0.5, "{:.2f}'".format(fwhmy),
             verticalalignment="center", rotation=-90, transform=ax1.transAxes)
    ax1.autoscale(axis="y", tight=True)
    ax1.set_xlabel("Intensity (relative)")
    handles, labels = ax0.get_legend_handles_labels()
    ax_lg.legend(handles, labels)

    ax0.set_yticklabels([])
    ax1.set_xticklabels([])
    ax0.set_yticks([])
    ax1.set_xticks([])
    gr['lon'].set_ticks_visible(False)
    gr['lon'].set_ticklabel_visible(False)
    gr['lat'].set_ticks_visible(False)
    gr['lat'].set_ticklabel_visible(False)
    lat.set_major_formatter('m')
    lon.set_major_formatter('m')
    lon.set_ticks(spacing=10. * u.arcmin)
    lat.set_ticks(spacing=10. * u.arcmin)
    lon.set_ticks_position('b')
    lat.set_ticks_position('l')
    lon.set_axislabel('arcmin')
    lat.set_axislabel('arcmin')
    lon.grid(alpha=0, linestyle='solid')
    lat.grid(alpha=0, linestyle='solid')
    ax.text(50, 700, "{} \nFWMH major: {:.2f}' \nFWHM minor: {:.2f}'".format(helio_map.date.isot, fwhmx, fwhmy), color='w')

    gs.tight_layout(fig, rect=[0.05, 0.05, 0.95, 0.95])
    if save:
        plt.savefig(infits[:-5] + "_gauss_fit.png", dpi=400)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to fit a 2D gaussian in a LOFAR image')
    parser.add_argument('infits', help='input fits file')
    args = parser.parse_args()

    infits = args.infits
    model = infits.replace('image.fits', 'model.fits')
    conv_model = convolve_model(model)
    if not os.path.isfile(model.replace('model', 'convolved_model')):
        conv_model.save(model.replace('model', 'convolved_model'))

    helio_map = icrs_to_helio(model.replace('model', 'convolved_model'))
    resid_map = icrs_to_helio(infits.replace('image.fits', 'residual.fits'))
    # helio_map = icrs_to_helio(infits)
    xy_mesh = sunpy.map.all_coordinates_from_map(helio_map)
    xy_arcsec = [xy_mesh.Tx.arcsec, xy_mesh.Ty.arcsec]

    # Fitting stuff

    gmodel = Model(gauss_2d)
    params = make_params(helio_map, 10, 10, 0, 0.1)
    error = np.ones_like(np.ravel(helio_map.data)) * np.std(resid_map.data)#* 0.01 * np.max(helio_map.data)
    # error = np.ravel(resid_map.data)
    print("Beginning fit for " + infits)
    gfit = gmodel.fit(np.ravel(helio_map.data), params, xy=xy_arcsec, weights= 1/error)
    plot_nice(helio_map, gfit, save=True)
    # plt.close()
    # helio_resample = helio_map.resample([199, 199]*u.pix)
    # xy_mesh_resample = sunpy.map.all_coordinates_from_map(helio_resample)
    # xy_arcsec_resample = [xy_mesh_resample.Tx.arcsec, xy_mesh_resample.Ty.arcsec]
    # emcee_kws = dict(steps=1000, burn=20, thin=10, is_weighted=False,
    #                  progress=True)
    # emcee_params = gfit.params.copy()
    # emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))
    # gfit_mc = gmodel.fit(np.ravel(helio_resample.data), xy=xy_arcsec_resample,
    #                      params=emcee_params, method='emcee', fit_kws=emcee_kws)
    # plot_nice(helio_resample, gfit_mc)
    #
    # emcee_corner = corner.corner(gfit_mc.flatchain, labels=gfit_mc.var_names,
    #                              truths=list(gfit_mc.params.valuesdict().values()))

    df_dict = gfit.best_values.copy()
    for key in gfit.params.keys():
        df_dict[key+'_std'] = gfit.params[key].stderr
    df_dict['redchi'] = gfit.redchi
    df = pd.DataFrame(df_dict.values(), df_dict.keys(), columns=[helio_map.date.isot])

    pickle_path = "burst_properties_{}MHz_{}.pkl".format(int(np.round(helio_map.wavelength.value)), helio_map.date.isot[:10])

    if os.path.isfile(pickle_path):
        df0 = pd.read_pickle(pickle_path)
        df1 = pd.concat([df0,df], axis='columns')
        df1.to_pickle(pickle_path)
    else:
        df.to_pickle(pickle_path)
    # plt.show()
