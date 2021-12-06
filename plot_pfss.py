#!/usr/bin/env python

import argparse
import sys

from multiprocessing import Pool

import astropy.units as u
import colorcet as cc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pfsspy
import pfsspy.tracing as tracing
import sunpy.coordinates.frames as frames
import sunpy.io.fits
import sunpy.map

from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from sunpy import config
from sunpy.io.special import srs
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.time import TimeRange

def main(t):
    print("running for time {}".format(df.index[t]))
    trange = TimeRange(Time(df.index[t])-15*u.min,Time(df.index[t])+15*u.min)
    #this will have to do for now. Should probably glob or something
    burst_fits = 'vis_fits/30MHz/{}/good_times/fit_on_2021-11-30_visibility_fit_recreated_map_30MHz_{}.fits'.format(
        df.index[t][:10].replace('-', '_'), df.index[t])
    burst_map = sunpy.map.Map(burst_fits)
    # x0 = Angle(df['x0'][0]*u.rad)
    # y0 = Angle(df['y0'][0]*u.rad)
    burst_coord = df['burst_centre_coord'][t]
    #a.Time('2019/04/14 12:00','2019/04/14 12:01'),
    aia_result = Fido.search(a.Time(trange, near=Time(df.index[t])),
                          a.Instrument.aia, a.Physobs.intensity,
                          a.Wavelength(193*u.angstrom))
    gong_result = Fido.search(a.Time(trange, near=Time(df.index[t])),
                              a.Instrument.gong)
    srs_result = Fido.search(a.Time(trange, near=Time(df.index[t])),
                             a.Instrument.srs_table)
    # srs_res = srs_q[0][[0,4,8]]
    srs_dl = Fido.fetch(srs_result[0][0], overwrite=False)
    aia_dl = Fido.fetch(aia_result[0][0], overwrite=False)
    if len(gong_result[0]) == 1:
        gong_dl = Fido.fetch(gong_result[0][0], overwrite=False)
    else:
        try:
            trange = TimeRange(Time(df.index[t]) - 30 * u.min, Time(df.index[t]) + 30 * u.min)
            gong_result = Fido.search(a.Time(trange, near=Time(df.index[t])),
                                      a.Instrument.gong)
            gong_dl = Fido.fetch(gong_result[0][0], overwrite=False)
        except IndexError:
            print('No GONG result for time {}'.format(df.index[t]))
            return
    srs_table = srs.read_srs(srs_dl[0])
    srs_table = srs_table[srs_table['Number'] == 12738]
    if len(srs_table) == 1:
        srs_lat = Quantity(srs_table['Latitude'].data*srs_table['Latitude'].unit)#srs_table['Latitude']
        srs_lon = Quantity(srs_table['Longitude'].data*srs_table['Longitude'].unit)#srs_table['Longitude']
        ar_coord = SkyCoord(srs_lon, srs_lat, frame="heliographic_stonyhurst", obstime=srs_table.meta['issued'], observer="earth")
        ar_helioprojective = ar_coord.transform_to(frames.Helioprojective)
    else:
        ar_helioprojective = SkyCoord(-900*u.arcsec, 145*u.arcsec, frame = frames.Helioprojective)
        #about where the ar rotates onto disk on 2019-09-07
    if len(aia_dl.errors) == 0:
        aia = sunpy.map.Map(aia_dl)
    else:
        print("No AIA map loaded for time {}".format(df.index[t]))
        return aia_dl, aia_result
    gong_map = sunpy.map.Map(gong_dl)
    dtime = aia.date
    nrho = 100
    rss = 2.5
    print("calculating for time {}".format(df.index[t]))
    pfss_in = pfsspy.Input(gong_map, nrho, rss)
    m = pfss_in.map
    if np.abs(ar_helioprojective.Tx.value) < 700:
        hp_lon = np.linspace(ar_helioprojective.Tx.value-200, ar_helioprojective.Tx.value+200, 15) * u.arcsec
        hp_lat = np.linspace(ar_helioprojective.Ty.value-200, ar_helioprojective.Ty.value+200, 15) * u.arcsec
    else:
        hp_lon = np.linspace(-850, ar_helioprojective.Tx.value + 400, 15) * u.arcsec #hacky trick for when AR is just on the limb
        hp_lat = np.linspace(ar_helioprojective.Ty.value - 200, ar_helioprojective.Ty.value + 250, 15) * u.arcsec
    #Make a 2D grid from these 1D points
    lon, lat = np.meshgrid(hp_lon, hp_lat)
    seeds = SkyCoord(lon.ravel(), lat.ravel(),
                     frame=aia.coordinate_frame)
    # fig = plt.figure(figsize=(7,7))
    # ax = plt.subplot(projection=aia)
    # aia.plot(axes=ax)
    # ax.plot_coord(seeds, color='white', marker='o', linewidth=0)
    pfss_out = pfsspy.pfss(pfss_in)
    tracer = tracing.FortranTracer()
    try:
        flines = tracer.trace(seeds, pfss_out)
    except IndexError:
        print("PFSS error for time {}".format(df.index[t]))
        return aia, None
    print("plotting for time {}".format(df.index[t]))
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot(1, 1, 1, projection=aia)
    with frames.Helioprojective.assume_spherical_screen(aia.observer_coordinate):
        aia.plot(ax, vmin = 10)
        aia.draw_grid()
        for fline in flines:
            if fline.is_open:
                ax.plot_coord(fline.coords, alpha=0.8, linewidth=1, color='yellow')
            else:
                ax.plot_coord(fline.coords, alpha=0.8, linewidth=1, color='white')
        # ax.plot_coord(burst_coord.transform_to(frames.HeliographicStonyhurst), color='r', marker='o')
        ax.set_facecolor('black')
        left_lim = aia.world_to_pixel(SkyCoord(-2500*u.arcsec,0*u.arcsec, frame=aia.coordinate_frame)).x.value
        right_lim = aia.world_to_pixel(SkyCoord(2500 * u.arcsec, 0 * u.arcsec, frame=aia.coordinate_frame)).x.value
        top_lim = aia.world_to_pixel(SkyCoord(0*u.arcsec,2500*u.arcsec, frame=aia.coordinate_frame)).y.value
        bottom_lim = aia.world_to_pixel(SkyCoord(0 * u.arcsec, -2500 * u.arcsec, frame=aia.coordinate_frame)).y.value
        ax.set_xlim(left_lim, right_lim)
        ax.set_ylim(bottom_lim, top_lim)
        ax.contour(burst_map.data, levels=np.arange(0.5, 1, 0.1) * np.max(burst_map.data),
                   transform=ax.get_transform(burst_map.wcs), cmap=cc.cm.CET_CBL3)
    # plt.savefig("./vis_fits/30MHz/pfss_{}.png".format(df.index[t]))
    # plt.close()
    return aia, flines
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pfile', help='Input pickle file. Must be output from plot_vis.py')
    parser.add_argument('-i', '--index', type=int,
                        help='Optional. Index of pickle file. Default `None`. Should replace this with time range at some point.')
    args = parser.parse_args()
    pfile = args.pfile
    i = args.index
    df = pandas.read_pickle(pfile)
    best_times_file = "best_times.txt" #"good_enough_times.txt"#
    best_times = np.loadtxt(best_times_file, dtype=str)
    df = df.loc[df.index.intersection(best_times)]
    if i is None:
        if len(df) == 0:
            print('Time not in list of good times')
        else:
            with Pool() as p:
                p.map(main, range(len(df)))
    else:
        aia, flines = main(i)
