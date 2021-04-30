#!/usr/bin/env python

import sys
import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u
infits =  sys.argv[1]
rot_deg = 30#float(sys.argv[2])
rot_angle = Angle(rot_deg*u.deg).rad
import pdb
def rotate_coords(x,y, theta):
    # rotate coordinates x y anticlockwise by theta
    x_p =  x*np.cos(theta) + y*np.sin(theta)
    y_p = -x*np.sin(theta) + y*np.cos(theta)
    return x_p,y_p


def fit_2d_gauss(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
        (x, y) = xy
        x, y = rotate_coords(x, y, theta)
        x0 = float(x0)
        y0 = float(y0)
        a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
        b = ((np.sin(2*theta))/(4*sig_x**2)) - ((np.sin(2*theta))/(4*sig_y**2))
        c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))
        # g = amp*np.exp(-(a*((x-x0)**2) + (2*b*(x-x0)*(y-y0)) + c*((y-y0)**2))) + offset
        g = amp*np.exp(-(((x-x0)**2)/(2*sig_x**2) + ((y-y0)**2)/(2*sig_y**2))) + offset
        return g.ravel()


def dirac_del_2d(xy, x0, y0):
    # pdb.set_trace()
    (x, y) = xy
    dd = np.zeros((len(x), len(y)))
    ind_x = np.where(x > x0)[1][0]
    ind_y = np.where(y > y0)[0][0]
    dd[ind_x,ind_y] = 1e10
    return dd

def FWHM_to_sig(FWHM):
    c = FWHM/(2*np.sqrt(2*np.log(2)))
    return c

with fits.open("../MS/20190404/models/13022887-t0001-dirty.fits") as psf:
    psf_data = psf[0].data[0,0]
    header = psf[0].header
#print(header)
with fits.open(infits, mode='update') as hdu:
    head_str = fits.Header.tostring(hdu[0].header)
    str_start = head_str.find('scale')
    str_end = head_str.find('asec')
    scale = Angle(float(head_str[str_start+6:str_end])*u.arcsec)
    xpix = hdu[0].header["NAXIS1"]
    ypix = hdu[0].header["NAXIS2"]
    x_cen = hdu[0].header['CRPIX1']
    y_cen = hdu[0].header['CRPIX2']
    x_cen = x_cen*scale
    y_cen = y_cen*scale
    sun_rad = Angle(0.25*u.deg)
    sun_rad_pix = sun_rad.deg/scale.deg

    #sigx = FWHM_to_sig(Angle(150*u.arcsec))#2*sun_rad)#_pix) #sun_rad_pix
    #sigy = FWHM_to_sig(Angle(300*u.arcsec))#2*sun_rad)#_pix) #sun_rad_pix

    sigx = FWHM_to_sig(Angle(10*u.arcmin))#2*sun_rad)#_pix) #sun_rad_pix
    sigy = FWHM_to_sig(Angle(15*u.arcmin))
    # sigx2 = FWHM_to_sig(Angle(5*u.arcmin))#.deg/scale.deg)
    # sigy2 = FWHM_to_sig(Angle(7*u.arcmin))#.deg/scale.deg)
    
    # sigx3 = FWHM_to_sig(Angle(0.06*u.deg))#.deg/scale.deg)
    # sigy3 = FWHM_to_sig(Angle(0.1*u.deg))#.deg/scale.deg)
    
    x = np.arange(xpix)*scale
    y = np.arange(ypix)*scale
    x = x - x[-1]/2
    y = y - y[-1]/2
    x0 = Angle(1000*u.arcsec)
    y0 = Angle(500*u.arcsec)
    # dpix =  1.39e-5
    # x = Angle(np.arange(-0.0142739,0.0142739,dpix)*u.rad)
    # y = Angle(np.arange(-0.0142739,0.0142739,dpix)*u.rad)

    xx, yy = np.meshgrid(x,y) 
    # gauss1 = fit_2d_gauss((xx.arcmin,yy.arcmin), 1, x_cen.arcmin, y_cen.arcmin, sigx.arcmin, sigy.arcmin, 0, 0)
    # gauss1 = gauss1.reshape(len(x), len(y))
    
    # gauss2 = fit_2d_gauss((xx.arcmin,yy.arcmin), 48453098.128, -21, -14.8, 3.25, 7.77, 123.74 , 17919.398 )
    # gauss2 = gauss2.reshape(len(x), len(y))

    # gauss3 = fit_2d_gauss((xx.arcmin,yy.arcmin), 3, 1250*scale.arcmin, 500*scale.arcmin, sigx3.arcmin, sigy3.arcmin, np.pi/3 , 0)
    # gauss3 = gauss3.reshape(len(x), len(y))
    
    gauss = fit_2d_gauss((xx.arcmin,yy.arcmin), 1, x0.arcmin, y0.arcmin, sigx.arcmin, sigy.arcmin, rot_angle, 0)
    gauss = gauss.reshape(len(x), len(y))
    dd = dirac_del_2d((xx.arcmin,yy.arcmin), x0.arcmin, y0.arcmin)
    # multi_gauss = gauss1+gauss2+gauss3
    hdu[0].data[0,0] = gauss
    hdu.flush()
