#!/usr/bin/env python

import sys
import sunpy
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import sunpy.coordinates.sun as sun
from math import cos, sin
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from sunpy import coordinates
from sunpy.map import header_helper
from sunpy.coordinates import frames
import pdb

def icrs_to_helio(smap):

    '''
    Note in the conversion from RA and DEC to helioprojective, astropy only handles this well 
    if the input coodinates are in true equinox pointing (which accounts for precession).
    In the LOFAR observations, the pointing is in J2000, which does not account for RA and DEC + precession. 
    Therefore, you have to convert from LOFAR J2000 to the true equinox to convert to Helioprojective. 
    Not sure how to do this at present.
    '''

    obstime = smap.meta['date-obs']
    dsun_obs = sun.earth_distance(obstime)
    latlon = [52.905329712, 6.867996528] * u.deg # LOFAR core in The Netherlands.
    gps = EarthLocation(lat=latlon[0], lon=latlon[1]) # LOFAR core in The Netherlands in ITRF.
    core_ITRF = np.array((3826577.066, 461022.948, 5064892.786))
    gps = EarthLocation.from_geocentric(*core_ITRF, u.m)
    lofar_coord = SkyCoord(gps.get_itrs(Time(obstime)))
    roll_angle = sunpy.coordinates.sun.P(obstime) #90*u.deg-sun.orientation(gps, obstime) #+ 270*u.deg
    crln_obs = sunpy.coordinates.sun.L0(obstime)
    crlt_obs = sunpy.coordinates.sun.B0(obstime)
    hgln_obs = sunpy.coordinates.sun.L0(obstime)*0.0
    hglt_obs = sunpy.coordinates.sun.B0(obstime)
    wavelnth = smap.meta['crval3']/1e6

    obs_coord = SkyCoord(smap.reference_coordinate,
            distance=dsun_obs,
            obstime=obstime,
            # frame='icrs',
            equinox='J2000').transform_to(frame=frames.Helioprojective(observer=lofar_coord), merge_attributes=True)

    pc = np.zeros([2,2])
    lamb = smap.meta['cdelt1']/smap.meta['cdelt2']
    pc[0,0] = cos(roll_angle.rad)
    pc[1,0] = sin(roll_angle.rad)/lamb
    pc[0,1] = -lamb*sin(roll_angle.rad)
    pc[1,1] = cos(roll_angle.rad)
    smap = smap.rotate(angle=-roll_angle)

    [sun_ra, sun_dec] = sun.sky_position(t=sunpy.time.parse_time(obstime),
        equinox_of_date=False) # False for J2000.
    suncoords = SkyCoord(sun_ra, sun_dec)

    delra = smap.reference_coordinate.ra - suncoords.ra
    deldec = smap.reference_coordinate.dec - suncoords.dec
    delpixx = delra/smap.scale.axis1
    delpixy = deldec/smap.scale.axis2
    crpix1 = smap.reference_pixel.x - delpixx
    crpix2 = smap.reference_pixel.y - delpixy

    smap_hdr = header_helper.make_fitswcs_header(
                smap.data,
                obs_coord,
                u.Quantity(smap.reference_pixel),
                u.Quantity(smap.scale))#,rotation_matrix=pc)

    smap_hdr.update({'cdelt1': abs(smap_hdr['cdelt1'])})
    smap_hdr.update({'crota': roll_angle.value})
    smap_hdr.update({'crln_obs': crln_obs})
    smap_hdr.update({'crlt_obs': crlt_obs})
    smap_hdr.update({'crpix1': crpix1.value})
    smap_hdr.update({'crpix2': crpix2.value})
    smap_hdr.update({'crval1': 0.0}) #arcsec
    smap_hdr.update({'crval2': 0.0}) #arcsec
    smap_hdr.update({'hgln_obs': hgln_obs.value})  # Corrected reference pixels here. Astropy transformation is wrong.
    smap_hdr.update({'hglt_obs': hglt_obs.value})
    smap_hdr.update({'crln_obs': crln_obs.value})
    smap_hdr.update({'crlt_obs': crlt_obs.value})
    smap_hdr.update({'dsun_obs': dsun_obs.meter})
    smap_hdr.update({'wavelnth': wavelnth})

    helio_map = sunpy.map.Map(smap.data, smap_hdr)

    return helio_map
    
