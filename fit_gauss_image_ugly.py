#!/usr/bin/env python

"""
Fit a gauss to a single burst in image space
Pearse Murphy 30/03/20 COVID-19
Takes fits file created by WSClean as input
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import sunpy
from sunpy.map import Map, all_coordinates_from_map
from sunpy.coordinates.frames import Helioprojective
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord, Latitude, Longitude 
from lmfit import Parameters, Model
import icrs_to_helio_old as icrs_to_helio
#import pdb
import warnings
warnings.filterwarnings("ignore")
def gauss_2d(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
	#create a 2D gaussian with input parameters
	#can't do this because it takes too long, assume it's been done outside function
	#	x = xy.Tx.arcsec
	#	y = xy.Ty.arcsec
	(x, y) = xy
	x0 = float(x0)
	y0 = float(y0)
	a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
	b = ((np.sin(2*theta))/(4*sig_x**2)) - ((np.sin(2*theta))/(4*sig_y**2))
	c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))
	g = amp*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))) + offset
	return g.ravel()

def pix_locs(smap):
	#return real world value of every pixel in smap
	xy_pix = np.indices(smap.data.shape)*u.pix
	xy_mesh = smap.pixel_to_world(xy_pix[0], xy_pix[1])
	return xy_mesh

def make_init_params(smap, fwhm_x, fwhm_y, theta, offset):

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
	zoom = smap.submap(bottom_left, top_right)
	rot = zoom.rotate(-theta)
	return rot
#loading stuff
#pdb.set_trace()
lofarfile = sys.argv[1]
lofarmap = Map(lofarfile)
lofarmap.plot_settings['cmap'] = 'viridis'

try:
    heliomap0 = icrs_to_helio.icrs_to_helio(lofarmap)
except KeyError:
    lofarmap.meta['date-obs'] = '2015-03-20T10:55:00.114'
    lofarmap.meta['crval3'] = 149017333.9777
    heliomap0 = icrs_to_helio.icrs_to_helio(lofarmap)

if lofarmap.dimensions.x.value >= 3000:

    max_xy = np.where(heliomap0.data == heliomap0.data.max())
    max_pos = heliomap0.pixel_to_world(max_xy[1][0]*u.pix, max_xy[0][0]*u.pix)
    xmax, ymax = max_pos.Tx, max_pos.Ty   
    bl = SkyCoord(xmax - 0.1*u.deg, ymax - 0.1*u.deg, frame = heliomap0.coordinate_frame)
    tr = SkyCoord(xmax + 0.1*u.deg, ymax + 0.1*u.deg, frame = heliomap0.coordinate_frame)

    heliomap0 = heliomap0.submap(bl, tr)
model = False #change to true to test a model source
#defining initial params
xy_mesh = all_coordinates_from_map(heliomap0)#pix_locs(heliomap0).T
xy_arcsec = [xy_mesh.Tx.arcsec, xy_mesh.Ty.arcsec]
#Fitting stuff
gmodel = Model(gauss_2d)
if model:
    model_gauss = gauss_2d(xy_arcsec,2000,-300,50,
    			Angle(9*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
    			Angle(19*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
    			0.5,10)
   # model_gauss = gauss_2d(xy_arcsec,2700,-587,130,
   # 			Angle(5*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
   # 			Angle(14*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
   # 			0.2,110)
    noise = 0.06
    model_gauss = model_gauss + noise*model_gauss.max()*np.random.normal(size=model_gauss.shape)
    heliomap = sunpy.map.Map(model_gauss.reshape(heliomap0.data.shape), heliomap0.meta)
else:
    heliomap = heliomap0
if len(sys.argv) > 2:
    #rude and crude implementation of setting initial parameters. Sorry.
    fwhm_x0, fhwm_y0, theta0, offset0 = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    params = make_params(heliomap, fwhm_x0, fhwm_y0, theta0, offset0)
else:
    params = make_params(heliomap)
print("Beginning fit for "+lofarfile)
gfit = gmodel.fit(np.ravel(heliomap.data), params, xy=xy_arcsec)
#this takes longer than I would like something to do with it not being a np.meshgrid?
heliomap.plot_settings['cmap'] = 'viridis'
#Preparing stuff for a pretty plot
x0 = gfit.params['x0'] * u.arcsec
y0 = gfit.params['y0'] * u.arcsec
theta = Angle(gfit.params['theta'] * u.rad)
gauss_centre = Helioprojective(x0,y0, observer='earth', obstime=heliomap0.date)
fit_map = sunpy.map.Map(gfit.best_fit.reshape(heliomap.data.shape), heliomap0.meta)

rot_fit = rotate_zoom(fit_map, x0, y0, theta) #fit_zoom.rotate(-theta)
rot_helio = rotate_zoom(heliomap, x0, y0, theta)#helio_zoom.rotate(-theta)

zoom_centre = rot_helio.world_to_pixel(gauss_centre)
zoom_xy = all_coordinates_from_map(rot_helio) #pix_locs(rot_helio)
x_cen = int(zoom_centre.x.round().value)
y_cen = int(zoom_centre.y.round().value)
new_dims = [100,100]*u.pix  #Resample data to 100 * 100 for histogram plot
helio_resample = rot_helio.resample(new_dims)
resample_xy = all_coordinates_from_map(helio_resample)#pix_locs(helio_resample)
#take a slice through the middle index, 49. Should do this properly at somepoint.
x_1D_helio, y_1D_helio =  helio_resample.data[49,:], helio_resample.data[:,49]
x_1D_fit, y_1D_fit =  rot_fit.data[y_cen,:], rot_fit.data[:,x_cen]
#x_1D_hist = np.histogram(x_1D_helio, nbins)
#y_1D_hist = np.histogram(y_1D_helio, nbins)
zoom_xarr = zoom_xy[y_cen, :]#zoom_xy.Tx[0]
zoom_yarr = zoom_xy[:, x_cen]#zoom_xy.Ty.T[0]
resample_xarr = resample_xy[49,:]
resample_yarr = resample_xy[:, 49]
coord_x = rot_helio.pixel_to_world([0,(zoom_xy.shape[1]-1)]*u.pix, [y_cen, y_cen]*u.pix)
coord_y = rot_helio.pixel_to_world([x_cen, x_cen]*u.pix, [0,(zoom_xy.shape[0]-1)]*u.pix)
#Printing stuff
print(gfit.fit_report())
fwhmx = Angle((2*np.sqrt(2*np.log(2))*gfit.params['sig_x']) * u.arcsec).arcmin
fwhmy = Angle((2*np.sqrt(2*np.log(2))*gfit.params['sig_y']) * u.arcsec).arcmin
print(fwhmx, fwhmy)

hwhmx_pixels = fwhmx*30*u.arcsec/rot_helio.scale.axis1
hwhmy_pixels = fwhmy*30*u.arcsec/rot_helio.scale.axis2
coord_x_hwhml = rot_helio.pixel_to_world([x_cen, (zoom_xy.shape[1]-1)]*u.pix, [y_cen-hwhmy_pixels.value, y_cen-hwhmy_pixels.value]*u.pix)
coord_x_hwhmr = rot_helio.pixel_to_world([x_cen, (zoom_xy.shape[1]-1)]*u.pix, [y_cen+hwhmy_pixels.value, y_cen+hwhmy_pixels.value]*u.pix)
coord_y_hwhml = rot_helio.pixel_to_world([x_cen - hwhmx_pixels.value, x_cen - hwhmx_pixels.value,]*u.pix, [y_cen, (zoom_xy.shape[0]-1)]*u.pix)
coord_y_hwhmr = rot_helio.pixel_to_world([x_cen + hwhmx_pixels.value, x_cen + hwhmx_pixels.value,]*u.pix, [y_cen, (zoom_xy.shape[0]-1)]*u.pix)
beam_cen = [(x_cen - 200), (y_cen - 200)]
#Plotting stuff
#heliomap.plot(title="Burst at {} MHz {}".format(str(np.round(heliomap.wavelength.value,3)),heliomap.date.isot))
fig = plt.figure(figsize = (8, 8))
gs = GridSpec(4,4)
ax = fig.add_subplot(gs[1:4,0:3], projection = rot_helio)
ax0 = fig.add_subplot(gs[0:1,0:3])
ax1 = fig.add_subplot(gs[1:4,3:])
ax_lg = fig.add_subplot(gs[0:1,3])
ax_lg.axis('off')
helio_plot = rot_helio.plot(axes=ax, title='')
rot_helio.draw_limb(ax)
BMAJ, BMIN, BPA = [Angle(lofarmap.meta[key], 'deg') for key in ['bmaj','bmin','bpa']]
solar_PA = sunpy.coordinates.sun.P(lofarmap.date).deg
#patch is all in pixels. There's probably an easy way to get to WCS.
beam = Ellipse((beam_cen[0], beam_cen[1]), 
                (BMAJ/abs(lofarmap.scale.axis1)).value, (BMIN/abs(lofarmap.scale.axis2)).value,
                90-BPA.deg+solar_PA+theta.deg,
                color='w', ls='--', fill=False)
ax.add_patch(beam)
gr = rot_helio.draw_grid(ax)
rot_fit.draw_contours(axes=ax,levels=[50]*u.percent, colors=['red'])
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
ax0.axvline(gauss_centre.Tx.arcmin + fwhmx*0.5,color='grey')
ax0.axvline(gauss_centre.Tx.arcmin - fwhmx*0.5,color='grey')
#ax0.hlines(0.5*np.max(x_1D_fit), gauss_centre.Tx.arcmin - fwhmx*0.5, gauss_centre.Tx.arcmin + fwhmx*0.5,
#        color='grey', linestyles='dashed')
ax0.annotate("",xy=(gauss_centre.Tx.arcmin - fwhmx*0.5, 0.5*np.max(x_1D_fit)),xycoords="data",
        xytext=(gauss_centre.Tx.arcmin + fwhmx*0.5, 0.5*np.max(x_1D_fit)), textcoords="data",
        arrowprops=dict(arrowstyle="<->"))
ax0.text(0.5, 0.4,"{:.2f}'".format(fwhmx),
         horizontalalignment="center", transform=ax0.transAxes)
ax0.autoscale(axis="x",tight=True)
ax0.set_ylabel("Intensity (relative)")
#right plot
ax1.plot(y_1D_helio,resample_yarr.Ty.arcmin,drawstyle='steps-mid')#, label="LOFAR source")
ax1.plot(y_1D_fit, zoom_yarr.Ty.arcmin)#, label="Modelled Gaussian")
ax1.axhline(gauss_centre.Ty.arcmin + fwhmy*0.5,color='grey')
ax1.axhline(gauss_centre.Ty.arcmin - fwhmy*0.5,color='grey')
#ax1.vlines(0.5*np.max(y_1D_fit), gauss_centre.Ty.arcmin - fwhmy*0.5, gauss_centre.Ty.arcmin + fwhmy*0.5,
#        color='grey', linestyles='dashed')
ax1.annotate("",xy=(0.5*np.max(y_1D_fit),gauss_centre.Ty.arcmin - fwhmy*0.5),xycoords="data",
        xytext=(0.5*np.max(y_1D_fit),gauss_centre.Ty.arcmin + fwhmy*0.5), textcoords="data",
        arrowprops=dict(arrowstyle="<->"))
ax1.text(0.5,0.5,"{:.2f}'".format(fwhmy),
         verticalalignment="center", rotation=-90, transform=ax1.transAxes)
ax1.autoscale(axis="y",tight=True)
ax1.set_xlabel("Intensity (relative)")
handles, labels = ax0.get_legend_handles_labels()
ax_lg.legend(handles, labels)
#ax1.legend()#bbox_to_anchor=(1.5, 1.5))
ax0.set_yticklabels([])
#ax0.set_xticklabels([])
#ax1.set_yticklabels([])
#ax0.set_xticklabels(np.arange(-50, 20,10))
#ax1.set_yticklabels(np.arange(-40, 30,10))
ax1.set_xticklabels([])
ax0.set_yticks([])
#ax0.set_xticks([])
#ax1.set_yticks([])
#ax0.set_xticks(np.arange(-50, 20,10)*60)
#ax1.set_yticks(np.arange(-40, 30,10)*60)
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
#lon.set_ticklabel_position('t')
#lat.set_ticklabel_position('r')
#lon.set_axislabel_position('t')
#lat.set_axislabel_position('r')
lon.set_axislabel('arcmin')#,minpad=0.0)
lat.set_axislabel('arcmin')#,minpad=0.0)
lon.grid(alpha=0, linestyle='solid')
lat.grid(alpha=0, linestyle='solid')
#lon.set_ticks(ax0.xaxis.get_majorticklocs()*u.arcmin)
#lon.set_ticklabel([*ax0.xaxis.get_majorticklabels()])
#lat.set_ticks(ax1.yaxis.get_majorticklocs()*u.arcmin)
#lat.set_ticklabel([*ax1.yaxis.get_majorticklabels()])
ax.text(50,700, "FWMH major: {:.2f}' \nFWHM minor: {:.2f}'".format(fwhmx, fwhmy),color='w')

gs.tight_layout(fig,rect=[0.05,0.05,0.95,0.95])
# if model:
#     ax0.set_title("Model Fit")
#     plt.savefig("gauss_fit_model.png", dpi=400)
# else:
#     #ax0.set_title("Data Fit") 
#     #plt.savefig(lofarfile[:-5]+"_gauss_fit.png", dpi=400)
    # plt.savefig("gauss_fit_data.png", dpi=400)
plt.show()

