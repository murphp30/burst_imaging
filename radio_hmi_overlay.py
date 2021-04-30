#!/usr/bin/env python

"""
Input LOFAR image fits file and return overlay of
radio image on top of HMI magnetogram
"""

import argparse

import astropy.units as u
import cmocean
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map

from sunpy.net import Fido, attrs as a

from icrs_to_helio import icrs_to_helio

parser = argparse.ArgumentParser(description='Produce radio image overlay')
parser.add_argument('fits_file', help='Name of LOFAR image fits file.')
parser.add_argument('-o', '--output', default=None, help='Name of output file.')
args = parser.parse_args()

fits_file = args.fits_file
output = args.output
if output is None:
    output = fits_file[:-5]+'_hmi_overlay.png'
icrs_map = sunpy.map.Map(fits_file)
helio_map = icrs_to_helio(fits_file)#icrs_to_helio(icrs_map)
mask = np.ma.masked_less(helio_map.data, np.max(helio_map.data)/5)
helio_map.mask = mask.mask

results = Fido.search(a.Time(helio_map.date-1*u.min, helio_map.date+1*u.min, near=helio_map.date),
                      a.Instrument.hmi, a.Physobs.los_magnetic_field)
hmi_download = Fido.fetch(results, path='./HMI/data', overwrite=False)
hmi_map = sunpy.map.Map(hmi_download)

helio_map.plot_settings['cmap'] = cmocean.cm.solar
comp_map = sunpy.map.Map(hmi_map, helio_map, composite=True)
lmax = (helio_map.data).max()
levels = lmax*np.arange(0.5, 1.1, 0.1)

comp_map.set_levels(index=1, levels=levels)
hmi_norm = matplotlib.colors.Normalize(vmin=-100, vmax=100)
plot_settings_dict = comp_map.get_plot_settings()[0]
plot_settings_dict['norm'] = hmi_norm
lofar_norm = matplotlib.colors.Normalize(vmin=np.percentile(helio_map.data, 33.3), 
                                         vmax=np.max(helio_map.data))
plot_settings_dict0 = comp_map.get_plot_settings()[1]
plot_settings_dict0['norm'] = lofar_norm
comp_map.set_plot_settings(1, plot_settings_dict0)
comp_map.set_plot_settings(0, plot_settings_dict)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)#(projection=hmi_map)
axlims = [-2000,2000]
# hmi_map.plot(ax)
comp_map.plot(ax, title='')#{} {}'.format(helio_map.date.isot[:-4], np.round(helio_map.wavelength,2)))
ax.set_xlim(axlims)
ax.set_ylim(axlims)
ax.text(-1900,1750, 'LOFAR: {} {}\nHMI: {}'.format(np.round(helio_map.wavelength,2),helio_map.date.isot[:-4],hmi_map.date.isot[:-4]))
# plt.colorbar()
# plt.tight_layout()
plt.savefig(output)
plt.close()
#plt.show()
