#!/usr/bin/env python
# Plots fits file generated by WSClean
import argparse
import glob
import os
import warnings

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import sunpy.map

from multiprocessing import Pool

from astropy.coordinates import Angle, SkyCoord
from mpl_toolkits.axes_grid1 import make_axes_locatable

from manual_clean import convolve_model
from icrs_to_helio import icrs_to_helio
from radec_to_hpc_map import radec_to_hpc

warnings.filterwarnings("ignore")


def plot_fits(fits_in, out_png=None):
    if out_png is None:
        out_png = fits_in[:-4] + 'png'
    smap = sunpy.map.Map(fits_in)
    smap.meta['wavelnth'] = smap.meta['crval3'] / 1e6
    smap.meta['waveunit'] = "MHz"
    fig = plt.figure()
    helio_smap = icrs_to_helio(fits_in)
    solar_PA = sunpy.coordinates.sun.P(smap.date).deg
    if fits_in[-8:] != "psf.fits":
        ax0 = fig.add_subplot(1, 1, 1, projection=helio_smap)
        helio_smap.plot_settings["title"] = str(
            np.round(helio_smap.meta['wavelnth'], 2)) + " MHz " + helio_smap.date.isot
        helio_smap.plot(cmap='viridis')
        if fits_in[-10:] != "model.fits":
            helio_smap.draw_limb(color='r')

        b = Ellipse((200, 200), Angle(smap.meta['BMAJ'] * u.deg).arcsec / abs(smap.scale[0].to(u.arcsec / u.pix).value),
                    Angle(smap.meta['BMIN'] * u.deg).arcsec / abs(smap.scale[1].to(u.arcsec / u.pix).value),
                    angle=(90 + smap.meta['BPA']) - solar_PA, fill=False, color='w', ls='--')
    else:
        ax0 = fig.add_subplot(1, 1, 1, projection=smap)
        smap.plot()
        b = Ellipse((smap.reference_pixel[0].value, smap.reference_pixel[1].value),
                    Angle(smap.meta['BMAJ'] * u.deg).arcsec / abs(smap.scale[0].to(u.arcsec / u.pix).value),
                    Angle(smap.meta['BMIN'] * u.deg).arcsec / abs(smap.scale[1].to(u.arcsec / u.pix).value),
                    angle=90 + smap.meta['BPA'], fill=False, color='w', ls='--')
    ax0.add_patch(b)
    plt.colorbar()
    print("Saving to {}".format(out_png))
    plt.savefig(out_png)
    plt.close()


def plot_compare(fits_in, out_png=None):
    if out_png is None:
        out_png = fits_in[:-10] + 'compare.png'
    if fits_in[-10:] != "image.fits":
        print("Input fits must be clean image. e.g. wsclean-image.fits")
        return
    dirty_fits = fits_in[:-10] + "dirty.fits"
    helio_clean = icrs_to_helio(fits_in)
    helio_dirty = icrs_to_helio(dirty_fits)
    icrs_map = sunpy.map.Map(fits_in)
    solar_PA = sunpy.coordinates.sun.P(icrs_map.date).deg
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for smap, ax, label in zip([helio_dirty, helio_clean], ax, ['Dirty', 'Clean']):
        smap.plot_settings['title'] = str(np.round(smap.meta['wavelnth'], 2)) + " MHz " + smap.date.isot
        smap.plot_settings['cmap'] = 'viridis'
        # smap.plot_settings['norm'] = matplotlib.colors.Normalize(vmin=0, vmax=500)
        im = smap.plot(axes=ax)
        smap.draw_limb(color='r')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.text(-3000, 3000, label, fontdict={'fontsize': 14, 'color': 'white'})

        b = Ellipse((-3000, -3000),
                    Angle(icrs_map.meta['BMAJ'] * u.deg).arcsec / abs(icrs_map.scale[0].to(u.arcsec / u.pix).value),
                    Angle(icrs_map.meta['BMIN'] * u.deg).arcsec / abs(icrs_map.scale[1].to(u.arcsec / u.pix).value),
                    angle=(90 + icrs_map.meta['BPA']) - solar_PA, fill=False, color='w', ls='--')
        ax.add_patch(b)
    plt.tight_layout()
    print("Saving to {}".format(out_png))
    plt.savefig(out_png)
    plt.close()


def plot_grid(fits_in, out_png=None):
    if out_png is None:
        out_png = fits_in[:-10] + 'grid_compare.png'
    if fits_in[-10:] != "image.fits":
        print("Input fits must be clean image. e.g. wsclean-image.fits")
        return
    dirty_fits = fits_in[:-10] + "dirty.fits"
    residual_fits = fits_in[:-10] + "residual.fits"
    model_fits = fits_in[:-10] + "model.fits"

    helio_clean = radec_to_hpc(fits_in)#icrs_to_helio(fits_in)
    helio_dirty = radec_to_hpc(fits_in)#icrs_to_helio(dirty_fits)
    conv_model = radec_to_hpc(fits_in)#convolve_model(model_fits)
    #if not os.path.isfile(model_fits.replace('model.fits', 'convolved_model.fits')):
    conv_model.save(model_fits.replace('model.fits', 'convolved_model.fits'), overwrite=True)
    helio_residual = radec_to_hpc(fits_in)#icrs_to_helio(residual_fits)
    helio_model = radec_to_hpc(model_fits.replace('model.fits', 'convolved_model.fits'))#icrs_to_helio(model_fits.replace('model.fits', 'convolved_model.fits'))

    icrs_map = sunpy.map.Map(fits_in)
    # solar_PA = sunpy.coordinates.sun.P(icrs_map.date).deg
    # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection=helio_clean)
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1, projection=helio_clean)
    ax3 = fig.add_subplot(2, 2, 3, sharex=ax1, projection=helio_clean)
    ax4 = fig.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1, projection=helio_clean)

    for smap, a, label in zip([helio_dirty, helio_clean, helio_residual, helio_model], [ax1, ax2, ax3, ax4],
                              ['Dirty', 'Clean', 'Residual', 'Model']):
        smap.plot_settings['title'] = str(np.round(smap.meta['wavelnth'], 2)) + " MHz " + smap.date.isot
        smap.plot_settings['cmap'] = 'viridis'
        # smap.plot_settings['norm'] = matplotlib.colors.Normalize(vmin=0, vmax=500)
        im = smap.plot(axes=a, title='')
        smap.draw_limb(color='r')
        sun_centre = sunpy.coordinates.sun.sky_position(t=smap.date, equinox_of_date='J2000')
        sun_centre_coord = SkyCoord(sun_centre[0], sun_centre[1], sunpy.coordinates.sun.earth_distance(smap.date))
        # smap.draw_grid()
        # divider = make_axes_locatable(a)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(im, cax=cax)
        # a.text(-3000, 2900, label, fontdict={'fontsize': 12, 'color': 'white'})

        # b = Ellipse((-2500, -2500),
        #             Angle(icrs_map.meta['BMAJ'] * u.deg).arcsec, #/ abs(icrs_map.scale[0].to(u.arcsec / u.pix).value),
        #             Angle(icrs_map.meta['BMIN'] * u.deg).arcsec, # / abs(icrs_map.scale[1].to(u.arcsec / u.pix).value),
        #             angle=(90 + icrs_map.meta['BPA']) - solar_PA, fill=False, color='w', ls='--')
        # a.add_patch(b)
        a.plot_coord(sun_centre_coord, '+', color='white')
        a.set_ylabel('Solar Y (arcsec)')
        a.set_xlabel('Solar X (arcsec)')

    # plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.setp(ax2.get_xticklabels(), visible=False)
    # plt.setp(ax3.get_yticklabels(), visible=False)
    # plt.setp(ax4.get_yticklabels(), visible=False)
    fig.suptitle(helio_clean.date.isot)
    # plt.tight_layout()
    print("Saving to {}".format(out_png))
    # plt.savefig(out_png)
    # plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot a fits file generated by WSClean')
    parser.add_argument('fits_in', help='name of input fits file')
    parser.add_argument('-o', dest='out_png', help='name of output png.\
                        If not given defaults to same prefix as fits_in.',
                        default=None)
    parser.add_argument('-c', '--compare',
                        help='plot comparison between dirty and clean images.\
                        fits_in must be image.fits and have matching dirty.fits.',
                        action='store_true')
    parser.add_argument('-a', '--all',
                        help='plot all images of that match given wildcard.',
                        type=str, default=None)
    parser.add_argument('-g', '--grid',
                        help='plot comparison between dirty, clean, residuals and model.\
                        fits_in must be image.fits.',
                        action='store_true')
    args = parser.parse_args()
    fits_in = args.fits_in
    out_png = args.out_png
    compare = args.compare
    grid = args.grid
    plot_all = args.all
    if plot_all is None:
        if compare:
            plot_compare(fits_in, out_png)
        elif grid:
            plot_grid(fits_in, out_png)
        else:
            plot_fits(fits_in, out_png)
    else:
        mpl.use('Agg')
        all_files = glob.glob(plot_all)
        if len(all_files) == 0:
            print("No files found, exiting")

        all_files.sort()
        if compare:
            with Pool() as p:
                p.map(plot_compare, all_files)
        elif grid:
            with Pool() as p:
                p.map(plot_grid, all_files)
        else:
            with Pool() as p:
                p.map(plot_fits, all_files)
