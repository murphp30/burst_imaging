#!/usr/bin/env python

"""
Input LOFAR image fits file and return overlay of
radio image on top of AIA 171 
"""

import argparse
import os

import astropy.units as u
import cmocean
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map

from astropy.coordinates import Angle
from sunpy.coordinates import frames
from sunpy.net import Fido, attrs as a

from icrs_to_helio import icrs_to_helio
from manual_clean import convolve_model

parser = argparse.ArgumentParser(description='Produce radio image overlay')
parser.add_argument('fits_file', help='Name of LOFAR image fits file.')
parser.add_argument('-a', '--channel', type=int, help='Which AIA channel to plot default = 193', default=193)
parser.add_argument('-o', '--output', default=None, help='Name of output file.')
args = parser.parse_args()

fits_file = args.fits_file
output = args.output
channel = args.channel
if output is None:
    output = fits_file[:-5]+'_aia_overlay.png'
# icrs_map = sunpy.map.Map(fits_file)
model = fits_file.replace('image.fits', 'model.fits')
conv_model = convolve_model(model)
if not os.path.isfile(model.replace('model.fits', 'convolved_model.fits')):
    conv_model.save(model.replace('model.fits', 'convolved_model.fits'))

helio_map = icrs_to_helio(model.replace('model.fits', 'convolved_model.fits'))#icrs_to_helio(fits_file)
mask = np.ma.masked_less(helio_map.data, np.max(helio_map.data)/5)
helio_map.mask = mask.mask

results = Fido.search(a.Time(helio_map.date-1*u.min, helio_map.date+1*u.min, near=helio_map.date),
                      a.Instrument.aia, a.Physobs.intensity,
                      a.Wavelength(channel*u.angstrom))
aia_download = Fido.fetch(results, path='./AIA/data', overwrite=False)
aia_map = sunpy.map.Map(aia_download)

# for i, alpha in zip(range(len(aia_download)-1, -1, -1),np.linspace(0,1, len(aia_download)+1)[1:]):
# 	aia_map.set_alpha(i, alpha)
helio_map.plot_settings['cmap'] = cmocean.cm.haline
# aia_map.add_map(helio_map)
# comp_map = sunpy.map.Map(aia_map, helio_map, composite=True)
lmax = (helio_map.data).max()
levels = lmax*np.arange(0.5, 1.1, 0.1)

# aia_map.set_levels(index=-1, levels=levels)
# aia_norm = matplotlib.colors.Normalize(vmin=-100, vmax=100)
# plot_settings_dict = aia_map.get_plot_settings()[0]
# # plot_settings_dict['norm'] = aia_norm
# lofar_norm = matplotlib.colors.Normalize(vmin=np.percentile(helio_map.data, 33.3),
#                                          vmax=np.max(helio_map.data))
# plot_settings_dict0 = aia_map.get_plot_settings()[-1]
# plot_settings_dict0['norm'] = lofar_norm
# aia_map.set_plot_settings(-1, plot_settings_dict0)
# aia_map.set_plot_settings(0, plot_settings_dict)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection=aia_map)
with frames.Helioprojective.assume_spherical_screen(aia_map.observer_coordinate):
    aia_map.plot(title='')
    contours = ax.contour(helio_map.data,
               levels=levels,
               transform=ax.get_transform(helio_map.wcs), cmap=cmocean.cm.solar)
    ax.set_facecolor('black')

    ax.set_xlim(-2000,6000)
    ax.set_ylim(-2000,6000)
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111,projection=aia_map)

# # aia_map.plot(ax)
# aia_map.plot(ax, title='')#{} {}'.format(helio_map.date.isot[:-4], np.round(helio_map.wavelength,2)))

ax.text(-1900,-1750,
        'LOFAR: {} {}\nAIA {:.0f} {}: {}'.format(
            np.round(helio_map.wavelength,2),
            helio_map.date.isot[:-4],
            aia_map.wavelength.value,  r'$\AA$',
            aia_map.date.isot[:-4]),
        color = 'white'
        )
# fig.colorbar(contours, ax=ax, fraction=0.046, pad=0.02, label='Jy/beam')
# plt.colorbar(fraction=0.046, pad=0.02, label='Jy/beam')
# plt.tight_layout()
plt.savefig(output)
plt.close()
# plt.show()
