#!/usr/bin/env python
#determine scale parameter for WSClean.
#requires a file generated by
#msoverview in=MEASURMENT_SET >> ms_file
import sys
from astropy.constants import c
import astropy.units as u
from astropy.coordinates import Angle
with open(sys.argv[1]) as ms_file:

    ms_string = ms_file.read()

freq_s = ms_string.find("TOPO") + 4
freq_e = ms_string.find("195.312")

freq = float(ms_string[freq_s:freq_e])
l = c.value/(freq*1e6)
B = 84974.55079 #gotten manually from msoverview verbose
theta_deg = Angle((l/B)*u.rad)
theta_asec = theta_deg.arcsec
scale = round(theta_asec/4,4)
print(scale)
