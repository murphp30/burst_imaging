#!/usr/bin/env python

import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.coordinates.frames as frames
import sunpy.map

from astropy.coordinates import SkyCoord
from sunpy.io.special import srs
from sunpy.net import Fido, attrs as a

tr0 = a.Time('2019/04/04 12:00','2019/04/04 12:01')
tr1 = a.Time('2019/04/08 12:00','2019/04/08 12:01')
tr2 = a.Time('2019/04/12 12:00','2019/04/12 12:01')

results = []
for tr in [tr0, tr1, tr2]:
    q = Fido.search(a.Instrument.aia,
                    a.Physobs.intensity,
                    a.Wavelength(193*u.angstrom),
                    tr)
    results.append(q[0,0])

dls = Fido.fetch(*results, path='./AIA', overwrite=False)
dls.sort()
srs_tr = a.Time('2019/04/04 12:00', '2019/04/12 12:01')
srs_q = Fido.search(a.Instrument.srs_table, srs_tr)
srs_res = srs_q[0][[0,4,8]]
srs_dls = Fido.fetch(*srs_res, path='./AIA', overwrite=False)
srs_dls.sort()
ar_locs = []
for srs_dl in srs_dls:
    srs_table = srs.read_srs(srs_dl)
    srs_table = srs_table[srs_table['Number'] == 12738]
    lat = srs_table['Latitude']
    lon = srs_table['Longitude']
    ar_coord = SkyCoord(lon, lat, frame="heliographic_stonyhurst")
    ar_locs.append(ar_coord)

day0 = sunpy.map.Map(dls[0])
day1 = sunpy.map.Map(dls[1])
day2 = sunpy.map.Map(dls[2])

fig, axs = plt.subplots(1,3, figsize=(15,6), sharex=True, sharey=True)
for day, ax, ar_loc in zip([day0,day1,day2], axs, ar_locs):
    day.plot(axes=ax)

ax1.set_ylabel('')
ax2.set_ylabel('')

# fig = plt.subplots(figsize=(15,6))
# ax0 = plt.subplot(1,3,1, projection=day0)
# ax1 = plt.subplot(1,3,2, projection=day1, sharey=ax0)
# ax2 = plt.subplot(1,3,3, projection=day2, sharey=ax0)
# for day, ax, ar_loc in zip([day0,day1,day2], [ax0,ax1,ax2], ar_locs):
#     day.plot(axes=ax)
#     if len(ar_loc) > 0:
#         ax.plot_coord(ar_loc, 'o')


# plt.tight_layout()
# plt.savefig('/Users/murphp30/Documents/Postgrad/thesis/Images/aia193_ar_evolve.png')
plt.show()