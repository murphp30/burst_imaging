#!/usr/bin/env python

import sys
import pdb
import sunpy.map
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sunpy.coordinates import frames, sun
from sunpy.net import Fido, attrs as a
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.io import fits
from astropy.time import Time 

def icrs_to_helio(file):

    hdu = fits.open(file)
    header = hdu[0].header
    data = np.squeeze(hdu[0].data)
    obstime = header['date-obs'] # observational time
    freq = header['crval3']*u.Hz # frequency of observation

    # we need to get the observer location to convert to HPC coordinates. 
    # this is required, especially for when looking at obskects close to  the Moon
    # lofar_loc = EarthLocation.of_address('Exloo, The Netherlands')
    core_ITRF = np.array((3826577.066, 461022.948, 5064892.786))
    lofar_loc = EarthLocation.from_geocentric(*core_ITRF, u.m)
    lofar_coord = SkyCoord(lofar_loc.get_itrs(Time(obstime)))

    # reference coordinate from FITS file
    reference_coord = SkyCoord(header['crval1']*u.deg, header['crval2']*u.deg, 
                           frame='gcrs',
                           obstime=obstime, 
                           distance=sun.earth_distance(obstime),
                           equinox='J2000')
    
    reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=lofar_coord))


    # cdelt in arcsec rather than degrees
    cdelt1 = (np.abs(header['cdelt1'])*u.deg).to(u.arcsec)
    cdelt2 = (np.abs(header['cdelt2'])*u.deg).to(u.arcsec)


    P1 = sun.P(obstime)

    header_test = sunpy.map.make_fitswcs_header(data, reference_coord_arcsec, 
                                            reference_pixel=u.Quantity([header['crpix1'], header['crpix2']]*u.pixel),
                                            scale=u.Quantity([cdelt1, cdelt2]*u.arcsec/u.pix),
                                            rotation_angle=-P1,
                                            wavelength=freq.to(u.MHz), 
                                            observatory='LOFAR')



    lofar_map = sunpy.map.Map(data, header_test)
    lofar_map.meta['bmaj'] = header['BMAJ']
    lofar_map.meta['bmin'] = header['BMIN']
    lofar_map.meta['bpa'] = header['BPA']

    lofar_map = lofar_map.rotate()

    return lofar_map


# if __name__ == "__main__":

#     file = sys.argv[1]  #'/Users/eoincarley/Data/2013_may_31/lofar/fits_SB005_44MHz/L141641_SAP000_44MHz-t0115-image.fits'
#     helio_map = icrs_to_helio(file)

#     fig = plt.figure(figsize=(10,10))
#     ax = plt.subplot(projection=helio_map)
#     helio_map.plot(axes=ax,
#         cmap=plt.get_cmap('afmhot'),
#         vmin=np.percentile(helio_map.data, 60.0),
#         vmax=np.percentile(helio_map.data, 100.0))
#     helio_map.draw_grid(axes=ax, color='black')
#     helio_map.draw_limb(axes=ax, color='blue')
#     plt.show()