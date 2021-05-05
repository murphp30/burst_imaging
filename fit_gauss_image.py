#!/usr/bin/env python

"""
Fit a gauss to a single burst in image space
Pearse Murphy 30/03/20 COVID-19
Takes fits file created by WSClean as input
"""
import argparse
import pdb
import warnings

import astropy.units as u
import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt
import sunpy.map

from multiprocessing import Pool

from astropy.coordinates import Angle, SkyCoord, Latitude, Longitude, EarthLocation
from astropy.time import Time
from lmfit import Parameters, Model
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from sunpy.coordinates.frames import Helioprojective

from icrs_to_helio import icrs_to_helio

warnings.filterwarnings("ignore")
def gauss_2d(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
    """
    create a 2D gaussian with input parameters
    can't do this because it takes too long, assume it's been done outside function
    """
    #   x = xy.Tx.arcsec
    #   y = xy.Ty.arcsec
    (x, y) = xy
    x0 = float(x0)
    y0 = float(y0)
    a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
    b = ((np.sin(2*theta))/(4*sig_x**2)) - ((np.sin(2*theta))/(4*sig_y**2))
    c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))
    g = amp*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))) + offset
    return g.ravel()

def log_likelihood(params, xy, data, err):
    # amp, x0, y0, sig_x, sig_y, theta, offset, log_f = params
    amp, x0, y0, sig_x, sig_y, theta, offset = params

    model = gauss_2d(xy, amp, x0, y0, sig_x, sig_y, theta, offset)
    sigma2 = err ** 2 #+ model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((data - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(params, data, initial_params):
    # amp, x0, y0, sig_x, sig_y, theta, offset, log_f = params
    amp, x0, y0, sig_x, sig_y, theta, offset = params
    amp0, x00, y00, sig_x0, sig_y0, theta0, offset0 = initial_params

    if  0.75 * np.max(data) < amp < 1.1 * np.max(data) and x00 - 500.0 < x0 < x00 + 500.0 and y00 - 500.0 < y0 < y00 + 500.0 and \
    0.0 < sig_x < 2*sig_x0 and 0.0 < sig_y < 2*sig_y0 and 0.0 < theta < np.pi/2 and \
    0 < offset < 0.25 * np.max(data):# and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_probability(params, xy, data, err, initial_params):
    lp = log_prior(params,data, initial_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, xy, data, err)

def make_init_params(smap, fwhm_x, fwhm_y, theta, offset):
    '''
    make dictionary of initial parameters to model gauss_2d
    '''
    max_xy = np.where(smap.data == smap.data.max())
    max_pos = smap.pixel_to_world(max_xy[1][0]*u.pix, max_xy[0][0]*u.pix)
        #x and y positions are in the opposite places where you'd expect them and I don't
        #know why. This works so go with it.

    init_params = {"amp":smap.data.max(),
                   "x0":max_pos.Tx.arcsec,
                   "y0":max_pos.Ty.arcsec,
                   "sig_x":Angle(fwhm_x*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
                   "sig_y":Angle(fwhm_y*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
                   "theta":theta,
                   "offset":offset}
    return init_params 

def make_params(smap, fwhm_x=10, fwhm_y=18, theta=0.1, offset=0):
    '''
    create lmfit.Parameters object and set how much each can vary in model fitting
    '''
    init_params = make_init_params(smap, fwhm_x, fwhm_y, theta, offset)
    params = Parameters()
    params.add_many(("amp", init_params["amp"], True, 0.5*init_params["amp"], None),
                    ("x0", init_params["x0"], True, init_params["x0"] - 600, init_params["x0"] + 600),
                    ("y0", init_params["y0"], True, init_params["y0"] - 600, init_params["y0"] + 600),
                    ("sig_x", init_params["sig_x"], True, 0, 2*init_params["sig_x"]),
                    ("sig_y", init_params["sig_y"], True, 0, 2*init_params["sig_y"]),
                    ("theta", init_params["theta"], True, 0, np.pi),
                    ("offset", init_params["offset"], True, smap.data.min(),smap.data.max() ))
    return params

def rotate_zoom(smap, x0, y0,theta):
    #shift = smap.shift(x0, y0)
    top_right = SkyCoord( x0 + 2000 * u.arcsec, y0 + 2000 * u.arcsec, frame=smap.coordinate_frame)
    bottom_left = SkyCoord( x0 - 2000 * u.arcsec, y0 - 2000 * u.arcsec, frame=smap.coordinate_frame)
    zoom = smap.submap(bottom_left, top_right=top_right)
    rot = zoom.rotate(-theta)
    return rot

def plot_nice(smap, fit, save=True):
    smap.plot_settings['cmap'] = 'viridis'
    x0 = fit.best_values['x0'] * u.arcsec
    y0 = fit.best_values['y0'] * u.arcsec
    theta = Angle(fit.best_values['theta'] * u.rad)
    gauss_centre = SkyCoord(x0, y0, frame='helioprojective', observer=smap.observer_coordinate, obstime=smap.date)
    fit_map = sunpy.map.Map(fit.best_fit.reshape(smap.data.shape), smap.meta)
    rot_fit = rotate_zoom(fit_map, x0, y0, theta) 
    rot_helio = rotate_zoom(smap, x0, y0, theta)
    zoom_centre = rot_helio.world_to_pixel(SkyCoord(gauss_centre))
    zoom_xy = sunpy.map.all_coordinates_from_map(rot_helio)
    x_cen = int(zoom_centre.x.round().value)
    y_cen = int(zoom_centre.y.round().value)
    new_dims = [99,99]*u.pix  #Resample data to 100 * 100 for histogram plot
    helio_resample = rot_helio.resample(new_dims)
    resample_xy = sunpy.map.all_coordinates_from_map(helio_resample)



    #take a slice through the middle index, 49. Should do this properly at somepoint.
    x_1D_helio, y_1D_helio = helio_resample.data[49,:], helio_resample.data[:,49]
    x_1D_fit, y_1D_fit = rot_fit.data[y_cen,:], rot_fit.data[:,x_cen]
    zoom_xarr = zoom_xy[y_cen, :]
    zoom_yarr = zoom_xy[:, x_cen]
    resample_xarr = resample_xy[49,:]
    resample_yarr = resample_xy[:, 49]
    coord_x = rot_helio.pixel_to_world([0,(zoom_xy.shape[1]-1)]*u.pix, [y_cen, y_cen]*u.pix)
    coord_y = rot_helio.pixel_to_world([x_cen, x_cen]*u.pix, [0,(zoom_xy.shape[0]-1)]*u.pix)

    print(fit.fit_report())
    fwhmx = Angle((2*np.sqrt(2*np.log(2))*fit.best_values['sig_x']) * u.arcsec).arcmin
    fwhmy = Angle((2*np.sqrt(2*np.log(2))*fit.best_values['sig_y']) * u.arcsec).arcmin
    print('FWHMX: {} arcmin\nFWHMY: {} arcmin'.format(np.round(fwhmx,2), np.round(fwhmy,2)))

    hwhmx_pixels = fwhmx*30*u.arcsec/rot_helio.scale.axis1
    hwhmy_pixels = fwhmy*30*u.arcsec/rot_helio.scale.axis2
    coord_x_hwhml = rot_helio.pixel_to_world([x_cen, (zoom_xy.shape[1]-1)]*u.pix, [y_cen-hwhmy_pixels.value, y_cen-hwhmy_pixels.value]*u.pix)
    coord_x_hwhmr = rot_helio.pixel_to_world([x_cen, (zoom_xy.shape[1]-1)]*u.pix, [y_cen+hwhmy_pixels.value, y_cen+hwhmy_pixels.value]*u.pix)
    coord_y_hwhml = rot_helio.pixel_to_world([x_cen - hwhmx_pixels.value, x_cen - hwhmx_pixels.value,]*u.pix, [y_cen, (zoom_xy.shape[0]-1)]*u.pix)
    coord_y_hwhmr = rot_helio.pixel_to_world([x_cen + hwhmx_pixels.value, x_cen + hwhmx_pixels.value,]*u.pix, [y_cen, (zoom_xy.shape[0]-1)]*u.pix)
    beam_cen = [(x_cen - 200), (y_cen - 200)]
    # pdb.set_trace()
    #Plotting stuff
    fig = plt.figure(figsize = (8, 8))
    gs = GridSpec(4,4)
    ax = fig.add_subplot(gs[1:4,0:3], projection = rot_helio)
    ax0 = fig.add_subplot(gs[0:1,0:3])
    ax1 = fig.add_subplot(gs[1:4,3:])
    ax_lg = fig.add_subplot(gs[0:1,3])
    ax_lg.axis('off')
    helio_plot = rot_helio.plot(axes=ax, title='')
    rot_helio.draw_limb(ax)
    BMAJ, BMIN, BPA = [Angle(smap.meta[key], 'deg') for key in ['bmaj','bmin','bpa']]
    solar_PA = sunpy.coordinates.sun.P(smap.date).deg
    #patch is all in pixels. There's probably an easy way to get to WCS.
    beam = Ellipse((beam_cen[0], beam_cen[1]), 
                    (BMAJ/abs(smap.scale.axis1)).value, (BMIN/abs(smap.scale.axis2)).value,
                    90-BPA.deg+solar_PA+theta.deg,
                    color='w', ls='--', fill=False)
    ax.add_patch(beam)
    gr = rot_helio.draw_grid(ax)
    level = 100* (0.5*(fit.best_values['offset']+np.max(fit_map.data)))/np.max(fit_map.data)
    rot_fit.draw_contours(axes=ax,levels=[level]*u.percent, colors=['red'])
    lon = helio_plot.axes.coords[0]
    lat = helio_plot.axes.coords[1]
    ax.plot_coord(coord_x, '--', color='white')
    ax.plot_coord(coord_y, '--', color='white')
    ax.plot_coord(coord_x_hwhml, '-', color='grey')
    ax.plot_coord(coord_x_hwhmr, '-', color='grey')
    ax.plot_coord(coord_y_hwhml, '-', color='grey')
    ax.plot_coord(coord_y_hwhmr, '-', color='grey')
    
    #top plot
    ax0.plot(resample_xarr.Tx.arcmin,x_1D_helio,drawstyle='steps-mid', label="LOFAR source")
    ax0.plot(zoom_xarr.Tx.arcmin, x_1D_fit, label="Gaussian fit")
    ax0.axvline(coord_y_hwhml[0].Tx.arcmin, color='grey')
    ax0.axvline(coord_y_hwhmr[0].Tx.arcmin, color='grey')
    ax0.annotate("",
                 xy=(coord_y_hwhml[0].Tx.arcmin, 0.5*(fit.best_values['offset']+np.max(x_1D_fit))),
                 xycoords="data",
                 xytext=(coord_y_hwhmr[0].Tx.arcmin, 0.5*np.max(x_1D_fit)),
                 textcoords="data",
                 arrowprops=dict(arrowstyle="<->"))
    ax0.text(0.5, 0.4,"{:.2f}'".format(fwhmx),
             horizontalalignment="center", transform=ax0.transAxes)
    ax0.autoscale(axis="x",tight=True)
    ax0.set_ylabel("Intensity (relative)")

    #right plot
    ax1.plot(y_1D_helio,resample_yarr.Ty.arcmin,drawstyle='steps-mid')
    ax1.plot(y_1D_fit, zoom_yarr.Ty.arcmin)
    ax1.axhline(coord_x_hwhml[0].Ty.arcmin, color='grey')
    ax1.axhline(coord_x_hwhmr[0].Ty.arcmin,color='grey')
    ax1.annotate("",
                 xy=(0.5*(fit.best_values['offset']+np.max(y_1D_fit)),coord_x_hwhml[0].Ty.arcmin),
                 xycoords="data",
                 xytext=(0.5*np.max(y_1D_fit),coord_x_hwhmr[0].Ty.arcmin),
                 textcoords="data",
                 arrowprops=dict(arrowstyle="<->"))
    ax1.text(0.5,0.5,"{:.2f}'".format(fwhmy),
             verticalalignment="center", rotation=-90, transform=ax1.transAxes)
    ax1.autoscale(axis="y",tight=True)
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
    ax.text(50,700, "FWMH major: {:.2f}' \nFWHM minor: {:.2f}'".format(fwhmx, fwhmy),color='w')

    gs.tight_layout(fig,rect=[0.05,0.05,0.95,0.95])
    if save:
        plt.savefig(infits[:-5]+"_gauss_fit.png", dpi=400)
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to fit a 2D gaussian in a LOFAR image')
    parser.add_argument('infits', help='input fits file')
    args = parser.parse_args()

    infits = args.infits
    helio_map = icrs_to_helio(infits)
    xy_mesh = sunpy.map.all_coordinates_from_map(helio_map)
    xy_arcsec = [xy_mesh.Tx.arcsec, xy_mesh.Ty.arcsec]
    
    #Fitting stuff

    gmodel = Model(gauss_2d)
    params = make_params(helio_map, 9, 14, 0.1, 0)
    error = np.ones_like(np.ravel(helio_map.data)) *0.01*np.max(helio_map.data)
    print("Beginning fit for "+infits)
    gfit = gmodel.fit(np.ravel(helio_map.data), params, xy=xy_arcsec , weights=1/error)

    # nll = lambda *args: -log_likelihood(*args)
    # initial = [gfit.best_values[key] for key in gfit.best_values.keys()]
    # # initial.append(np.log(1))
    # # soln = minimize(nll, initial, args=(xy_arcsec, np.ravel(helio_map.data),error))
    # pos = initial * (1 + 1e-3 * np.random.randn(50,7))# + 1e-4 * np.random.randn(32,8)
    # nwalkers, ndim = pos.shape
    # # # with Pool() as pool:
    # #     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xy_arcsec, np.ravel(helio_map.data), error), pool=pool)
    # #     sampler.run_mcmc(pos, 5000, progress=True)


    # helio_small = helio_map.resample([100,100]*u.pix)
    # # xsmall = np.linspace(helio_map.bottom_left_coord.Tx, helio_map.top_right_coord.Tx, 100)
    # # ysmall = np.linspace(helio_map.bottom_left_coord.Ty, helio_map.top_right_coord.Ty, 100)
    # small_mesh = sunpy.map.all_coordinates_from_map(helio_small) #np.meshgrid(xsmall, ysmall)
    # small_xy_arcsec = [small_mesh.Tx.arcsec, small_mesh.Ty.arcsec]

    # # small_gauss = gauss_2d(small_xy_arcsec, *initial)
    # small_error = np.ones_like(np.ravel(helio_small.data)) *0.01*np.max(helio_small.data)
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(small_xy_arcsec, np.ravel(helio_small.data), small_error, initial))
    # sampler.run_mcmc(pos, 5000, progress=True)
 
    # fig, axes = plt.subplots(7, figsize=(10, 7), sharex=True)
    # samples = sampler.get_chain()                     
    # labels = [*gfit.best_values.keys()]
    # for i in range(ndim):
    #     ax = axes[i]
    #     ax.plot(samples[:, :, i], "k", alpha=0.3)
    #     ax.set_xlim(0, len(samples))
    #     ax.set_ylabel(labels[i])
    #     ax.yaxis.set_label_coords(-0.1, 0.5)
    
    # axes[-1].set_xlabel("step number")

    # flat_samples = sampler.get_chain(discard=100, thin=25, flat=True)



    # fig = corner.corner(flat_samples, labels=labels, truths=[*initial])
    # for i, param in enumerate(gfit.best_values.keys()):
    #     mcmc  = np.percentile(flat_samples[:,i], 50)
    #     print("Best {}: {}".format(param, mcmc))
    # with Pool() as pool:
    # emcee_kws = dict(steps=1000, burn=300, thin=20, workers, is_weighted=False)
    # emcee_params = gfit.params.copy()
    # emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2.))
    # mcfit = gmodel.fit(np.ravel(helio_map.data), params=emcee_params, xy=xy_arcsec, method='emcee',
    #                    fit_kws=emcee_kws)
    plot_nice(helio_map, gfit)
    plt.show()

