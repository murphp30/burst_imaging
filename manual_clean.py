#!/usr/bin/env python

"""Investigate issues with WSClean and remote stations."""
import argparse
import warnings

import astropy.units as u
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map

from astropy.coordinates import Angle, SkyCoord
from scipy.signal import convolve, fftconvolve
from matplotlib.patches import Ellipse

from make_gauss import rotate_coords, FWHM_to_sig

warnings.filterwarnings("ignore")

def make_map(fits_file):
    """Make sunpy map from fits file"""
    model_map = sunpy.map.Map(fits_file)
    model_map.meta['waveunit'] = "MHz"
    model_map.plot_settings['cmap'] = cmocean.cm.solar
    return model_map

def plot_model(fits_file, out=None, save=False):
    """Plot model image in icrs coordinates"""
    if out is None:
        out = fits_file[:-4] + 'png'

    model_map = make_map(fits_file)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1, projection=model_map)
    model_map.plot(axes=ax)
    b = Ellipse((model_map.reference_pixel[0].value, model_map.reference_pixel[1].value),
                Angle(model_map.meta['BMAJ'] * u.deg).arcsec / abs(model_map.scale[0].to(u.arcsec / u.pix).value),
                Angle(model_map.meta['BMIN'] * u.deg).arcsec / abs(model_map.scale[1].to(u.arcsec / u.pix).value),
                angle= -(90 - model_map.meta['BPA']), fill=False, color='w', ls='--', label='Beam FWHM')
    ax.add_patch(b)
    plt.legend()
    if save:
        plt.savefig(out)

def clean_beam(fits_file):
    """create a gaussian beam with parameters from fits file"""
    model_map = make_map(fits_file)
    bmaj = Angle(model_map.meta['BMAJ'] * u.deg)
    bmin = Angle(model_map.meta['BMIN'] * u.deg)
    bpa = Angle((model_map.meta['BPA']) * u.deg)
    pix_mesh = np.meshgrid(np.arange(model_map.dimensions[0].value), np.arange(model_map.dimensions[1].value), indexing='ij') * u.pix
    coord_arr = model_map.pixel_to_world(pix_mesh[0], pix_mesh[1])
    coord_mesh = np.zeros_like(pix_mesh.value)
    coord_mesh[0] = coord_arr.ra
    coord_mesh[1] = coord_arr.dec

    sig_maj = FWHM_to_sig(bmaj)
    sig_min = FWHM_to_sig(bmin)

    ra0 = model_map.reference_coordinate.ra
    dec0 = model_map.reference_coordinate.dec
    a = ((np.cos(bpa) ** 2) / (2 * sig_maj ** 2)) + ((np.sin(bpa) ** 2) / (2 * sig_min ** 2))
    b = ((np.sin(2 * bpa)) / (4 * sig_maj ** 2)) - ((np.sin(2 * bpa)) / (4 * sig_min ** 2))
    c = ((np.sin(bpa) ** 2) / (2 * sig_maj ** 2)) + ((np.cos(bpa) ** 2) / (2 * sig_min ** 2))
    beam = np.exp(-(a*((coord_arr.ra-ra0)**2) + (2*b*(coord_arr.ra-ra0)*(coord_arr.dec-dec0)) + c*((coord_arr.dec-dec0)**2)))
    beam_map = sunpy.map.Map((beam, model_map.meta))
    beam_map.plot_settings['cmap'] = cmocean.cm.solar
    return beam_map

def convolve_model(fits_file):
    model = make_map(fits_file)
    beam = clean_beam(fits_file)
    image = fftconvolve(beam.data, model.data,'same')#convolve(1*model.data, beam.data,'same', 'direct')
    image_map = sunpy.map.Map((image, model.meta))
    image_map.plot_settings['cmap'] = cmocean.cm.solar
    return image_map

def recreate_image(fits_file):
    model = convolve_model(fits_file)
    residuals = make_map(fits_file.replace('model.fits','residual.fits'))
    recreated_data = model.data + residuals.data
    recreated_map = sunpy.map.Map((recreated_data, model.meta))
    recreated_map.plot_settings['cmap'] = cmocean.cm.solar
    return recreated_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manually convolve input model image with clean beam.')
    parser.add_argument('model', help='name of input model fits file')
    args = parser.parse_args()
    model = args.model

    #plot_model(model)
    # model_map = make_map(model)
    # beam = clean_beam(model)
    # image = convolve_model(model)
    # plot_model(model.replace('model', 'psf'))

    # fig, ax = plt.subplots(3,1, figsize=(5,15), sharex=True)
    # model_map.plot(axes=ax[0], title='')
    # beam.plot(axes=ax[1], title='')
    # image.plot(axes=ax[2], title='')

    # for a, label in zip(ax, ['Model', 'Beam', 'Convolved Image']):
    #     a.text(13.6, 5.0, label, fontdict={'fontsize': 12, 'color': 'white'})
    # plt.tight_layout()

        #
        # plt.colorbar()

    # fig, ax1 = plt.subplots()
    # image.plot(axes=ax1,title='convolved')


    # fig, ax2 = plt.subplots()

    fig, ax = plt.subplots(3, 2, figsize=(10,10))

    for a, im in zip(ax.reshape(-1), ['dirty', 'image', 'psf', 'conv', 'residual', 'rec']):
        fname = model.replace('model.fits', im+'.fits')
        if im != 'conv' and im != 'rec':
            smap = make_map(fname)
        elif im == 'conv':
            smap = convolve_model(model)
        elif im == 'rec':
            smap = recreate_image(model)

        mplot = smap.plot(axes=a, title='')

        if im != 'psf':
            b = Ellipse((smap.top_right_coord.ra.deg + 0.2, smap.top_right_coord.dec.deg - 0.2),
                        (Angle(smap.meta['BMAJ'] * u.deg)).value,
                        (Angle(smap.meta['BMIN'] * u.deg)).value,
                        angle=(90 - smap.meta['BPA']), fill=False, color='w', ls='--')
            a.add_patch(b)
            fig.colorbar(mplot, ax=a, fraction=0.046, pad=0.02, label='Jy/beam')
        else:
            b = Ellipse((smap.reference_coordinate.ra.deg, smap.reference_coordinate.dec.deg),
                        (Angle(smap.meta['BMAJ'] * u.deg)).value,
                        (Angle(smap.meta['BMIN'] * u.deg)).value,
                        angle=(90 - smap.meta['BPA']), fill=False, color='w', ls='--')
            a.add_patch(b)
            fig.colorbar(mplot, ax=a, fraction=0.046, pad=0.02, label='')

    for a, label in zip(ax.reshape(-1), ['Dirty', 'Clean', 'PSF', 'Convolved Image', 'Residuals', 'Recreated Image']):
        a.text(smap.bottom_left_coord.ra.deg - 0.2, smap.bottom_left_coord.dec.deg + 0.2, label, fontdict={'fontsize': 12, 'color': 'white'})
    fig.suptitle(smap.date.isot)
    plt.savefig(model.replace('model.fits', 'compare.png'))
    plt.tight_layout()


    # conv_model = convolve_model(model)
    # residuals = make_map(model.replace('model', 'residual'))
    # clean = make_map(model.replace('model', 'image'))
    # rec = recreate_image(model)
    # clean_sub_resid = clean.data - residuals.data
    # clean_sub_rec = clean.data - rec.data
    # sub_data = clean_sub_resid - conv_model.data
    # plt.figure()
    # plt.imshow(clean_sub_resid, aspect='auto', origin='lower')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(conv_model.data, aspect='auto', origin='lower')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(clean_sub_resid, origin='lower',
    #            extent=[
    #                clean.bottom_left_coord.ra.deg,
    #                clean.top_right_coord.ra.deg,
    #                clean.bottom_left_coord.dec.deg,
    #                clean.top_right_coord.dec.deg
    #            ])
    # plt.xlabel('RA deg')
    # plt.ylabel('DEC deg')
    # plt.title('clean image - residuals')
    # plt.colorbar(fraction=0.046, pad=0.02, label='Jy/beam')
    # plt.tight_layout()
    # plt.savefig(model.replace('model.fits', 'cleannoresid.png'))

    # plt.figure()
    # plt.imshow(clean_sub_rec, origin='lower',
    #            extent=[
    #                clean.bottom_left_coord.ra.deg,
    #                clean.top_right_coord.ra.deg,
    #                clean.bottom_left_coord.dec.deg,
    #                clean.top_right_coord.dec.deg
    #            ])
    # plt.xlabel('RA deg')
    # plt.ylabel('DEC deg')
    # plt.title('Difference between recreated and clean image')
    # plt.colorbar(fraction=0.046, pad=0.02, label='Jy/beam')
    # plt.tight_layout()
    # plt.savefig(model.replace('model.fits', 'rec_diff.png'))
    # plt.show()


    plt.close('all')