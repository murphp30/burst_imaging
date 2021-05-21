#!/usr/bin/env python
#
# Authors:
# Joris van Zwieten
# Francesco de Gasperin
# Tammo Jan Dijkema
#
# Copyright (C) 2007
# ASTRON (Netherlands Institute for Radio Astronomy)
# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands
#
# This file is part of the LOFAR software suite.
# The LOFAR software suite is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The LOFAR software suite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
#
# $Id$

import sys, copy, math, numpy, signal
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import os.path
import string

import lofar.parmdb as parmdb

signal.signal(signal.SIGINT, signal.SIG_DFL)
__styles = ["%s%s" % (x, y) for y in ["-", ":"] for x in ["b", "g", "r", "c",
    "m", "y", "k"]]

def contains(container, item):
    try:
        return container.index(item) >= 0
    except ValueError:
        return False

def common_domain(parms):
    if len(parms) == 0:
        return None

    domain = [-1e30, 1e30, -1e30, 1e30]
    for parm in parms:
        tmp = parm.domain()
        domain = [max(domain[0], tmp[0]), min(domain[1], tmp[1]), max(domain[2], tmp[2]), min(domain[3], tmp[3])]

    if domain[0] >= domain[1] or domain[2] >= domain[3]:
        return None

    return domain

def unwrap(phase, tol=0.25, delta_tol=0.25):
    """
    Unwrap phase by restricting phase[n] to fall within a range [-tol, tol]
    around phase[n - 1].

    If this is impossible, the closest phase (modulo 2*pi) is used and tol is
    increased by delta_tol (tol is capped at pi).
    """

    assert(tol < math.pi)

    # Allocate result.
    out = numpy.zeros(phase.shape)

    # Effective tolerance.
    eff_tol = tol

    ref = phase[0]
    for i in range(0, len(phase)):
        delta = math.fmod(phase[i] - ref, 2.0 * math.pi)

        if delta < -math.pi:
            delta += 2.0 * math.pi
        elif delta > math.pi:
            delta -= 2.0 * math.pi

        out[i] = ref + delta

        if abs(delta) <= eff_tol:
            # Update reference phase and reset effective tolerance.
            ref = out[i]
            eff_tol = tol
        elif eff_tol < math.pi:
            # Increase effective tolerance.
            eff_tol += delta_tol * tol
            if eff_tol > math.pi:
                eff_tol = math.pi

    return out

def unwrap_windowed(phase, window_size=5):
    """
    Unwrap phase by estimating the trend of the phase signal.
    """

    # Allocate result.
    out = numpy.zeros(phase.shape)

    windowl = numpy.array([math.fmod(phase[0], 2.0 * math.pi)] * window_size)

    delta = math.fmod(phase[1] - windowl[0], 2.0 * math.pi)
    if delta < -math.pi:
        delta += 2.0 * math.pi
    elif delta > math.pi:
        delta -= 2.0 * math.pi
    windowu = numpy.array([windowl[0] + delta] * window_size)

    out[0] = windowl[0]
    out[1] = windowu[0]

    meanl = windowl.mean()
    meanu = windowu.mean()
    slope = (meanu - meanl) / float(window_size)

    for i in range(2, len(phase)):
        ref = meanu + (1.0 + (float(window_size) - 1.0) / 2.0) * slope
        delta = math.fmod(phase[i] - ref, 2.0 * math.pi)

        if delta < -math.pi:
            delta += 2.0 * math.pi
        elif delta > math.pi:
            delta -= 2.0 * math.pi

        out[i] = ref + delta

        windowl[:-1] = windowl[1:]
        windowl[-1] = windowu[0]
        windowu[:-1] = windowu[1:]
        windowu[-1] = out[i]

        meanl = windowl.mean()
        meanu = windowu.mean()
        slope = (meanu - meanl) / float(window_size)

    return out

def normalize(phase):
    """
    Normalize phase to the range [-pi, pi].
    """

    # Convert to range [-2*pi, 2*pi].
    out = numpy.fmod(phase, 2.0 * numpy.pi)

    # Remove nans
    numpy.putmask(out, out!=out, 0)
    # Convert to range [-pi, pi]
    out[out < -numpy.pi] += 2.0 * numpy.pi
    out[out > numpy.pi] -= 2.0 * numpy.pi

    return out

def plot(fig, y, x=None, clf=True, sub=None, scatter=False, stack=False,
    sep=5.0, sep_abs=False, labels=None, show_legend=False, title=None,
    xlabel=None, ylabel=None):
    """
    Plot a list of signals.

    If 'fig' is equal to None, a new figure will be created. Otherwise, the
    specified figure number is used. The 'sub' argument can be used to create
    subplots.

    The 'scatter' argument selects between scatter and line plots.

    The 'stack', 'sep', and 'sep_abs' arguments can be used to control placement
    of the plots in the list. If 'stack' is set to True, each plot will be
    offset by the mean plus sep times the standard deviation of the previous
    plot. If 'sep_abs' is set to True, 'sep' is used as is.

    The 'labels' argument can be set to a list of labels and 'show_legend' can
    be set to True to show a legend inside the plot.

    The min/max/median of the plotted points are returned is returned.
    """
    global __styles

    if clf:
        fig.clf()

    if sub is not None:
        fig.add_subplot(sub)

    axes = fig.gca()
    if not title is None:
        axes.set_title(title)
    if not xlabel is None:
        axes.set_xlabel(xlabel)
    if not ylabel is None:
        axes.set_ylabel(ylabel)

    if x is None:
        x = [range(len(yi)) for yi in y]

    offset = 0.
    med = 0.
    for i in range(0,len(y)):
        if labels is None:
            if scatter:
                axes.scatter(x[i], y[i] + offset, edgecolors="None",
                    c=__styles[i % len(__styles)][0], marker="o")
            else:
                axes.plot(x[i], y[i] + offset, __styles[i % len(__styles)])
        else:
            if scatter:
                axes.scatter(x[i], y[i] + offset, edgecolors="None",
                    c=__styles[i % len(__styles)][0], marker="o",
                    label=labels[i])
            else:
                axes.plot(x[i], y[i] + offset, __styles[i % len(__styles)],
                    label=labels[i])

        med += numpy.median(y[i] + offset)
        if stack and i != len(y)-1:
            if sep_abs:
                offset += sep
            else:
                offset += y[i].mean() + sep * y[i].std()

    fig.tight_layout()

    if not labels is None and show_legend:
        axes.legend(prop=FontProperties(size="x-small"), markerscale=0.5)

    # return min max median values
    if stack:
        return numpy.min(y[0]), numpy.max(y[-1]+offset), med/len(y)
    else:
        return numpy.min(y), numpy.max(y), med/len(y)

class Parm:
    """
    Each entry in the parmdb is an instance of this class
    """
    def __init__(self, db, name, elements=None, isPolar=True):
        self._db = db
        self._name = name
        self._elements = elements
        self._isPolar = isPolar
        self._value = None
        self._freq = None
        self._value_domain = None
        self._value_resolution = None
        self._calType = self._name.split(':')[0]

        if self._calType == 'DirectionalGain' or self._calType == 'RotationAngle': self._antenna = self._name.split(':')[-2]
        else: self._antenna = self._name.split(':')[-1]

        # modifiers for some phase outputs
        if self._calType == 'Clock':
            self.mod = lambda Clock: 2. * numpy.pi * Clock * self._freq
        elif self._calType == 'TEC':
            self.mod = lambda TEC: -8.44797245e9 * TEC / self._freq
        elif self._calType == 'RotationMeasure':
            self.mod = lambda RM: RM * (299792458.**2) / (self._freq)**2
        else:
            # everything else is plain
            self.mod = lambda ph: ph
            
        self._readDomain()

    def empty(self):
        return self._empty

    def domain(self):
        return self._domain

    def color(self):
        if self._calType == 'Gain': return QColor('#ffbfca')
        if self._calType == 'DirectionalGain': return QColor('#ebbfff')
        if self._calType == 'CommonRotationAngle': return QColor('#bfcdff')
        if self._calType == 'RotationAngle': return QColor('#bfe5ff')
        if self._calType == 'Clock': return QColor('#fff1bf')
        if self._calType == 'TEC': return QColor('#bfffda')
        if self._calType == 'CommonScalarPhase': return QColor('#ffbfbf')
        if self._calType == 'CommonScalarAmplitude': return QColor('#ffbfbf')
        if self._calType == 'ScalarPhase': return QColor('#ffbf00')
        if self._calType == 'ScalarAmplitude': return QColor('#ffbf00')
        if self._calType == 'RotationMeasure': return QColor('#84f0aa')
        return QColor('#FFFFFF')

    def valueAmp(self, domain=None, resolution=None, asPolar=True):
        self.updateValue(domain, resolution)

        if asPolar:
            if self._isPolar:
                ampl = self._value[0]
            else:
                ampl = numpy.sqrt(numpy.power(self._value[0], 2) + numpy.power(self._value[1], 2))

            return ampl

        if not self._isPolar:
            re = self._value[0]
        else:
            re = self._value[0] * numpy.cos(self._value[1])

        return re

    def valuePhase(self, domain=None, resolution=None, asPolar=True, unwrap_phase=False, \
            reference_parm=None, sum_parms=[], mod=True):
        self.updateValue(domain, resolution)

        if asPolar:
            if self._isPolar:
                phase = self._value[1]
            else:
                phase = numpy.arctan2(self._value[1], self._value[0])

            # apply modifiers for some solution types (e.g. clock, TEC, RM)
            phase = self.mod(phase)

            if not reference_parm is None:
                reference_val = reference_parm.valuePhase(domain, resolution)
                assert(reference_val.shape == phase.shape)
                phase = normalize(phase - reference_val)

            for sum_parm in sum_parms:
                sum_val = sum_parm[0].valuePhase(domain, resolution)
                sum_reference_val = sum_parm[1].valuePhase(domain, resolution)
                assert(sum_val.shape == phase.shape)
                assert(sum_reference_val.shape == phase.shape)
                phase = normalize(phase + sum_val - sum_reference_val)

            if mod: phase = normalize(phase)

            if unwrap_phase:
                for i in range(0, phase.shape[1]):
                    phase[:, i] = unwrap(phase[:, i])

            return phase

        if not self._isPolar:
            im = self._value[1]
        else:
            im = self._value[0] * numpy.sin(self._value[1])

        return im

    def _readDomain(self):
        if self._elements is None:
            self._domain = self._db.getRange(self._name)
        else:
            if self._elements[0] is None:
                self._domain = self._db.getRange(self._elements[1])
            elif self._elements[1] is None:
                self._domain = self._db.getRange(self._elements[0])
            else:
                domain_el0 = self._db.getRange(self._elements[0])
                domain_el1 = self._db.getRange(self._elements[1])
                self._domain = [max(domain_el0[0], domain_el1[0]), min(domain_el0[1], domain_el1[1]), max(domain_el0[2], domain_el1[2]), min(domain_el0[3], domain_el1[3])]

        self._empty = (self._domain[0] >= self._domain[1]) or (self._domain[2] >= self._domain[3])

    def updateValue(self, domain=None, resolution=None):
        if self.empty():
            return (numpy.zeros((1,1)), numpy.zeros((1,1)))

        # if nothig changes, use cache
        if self._value is None or self._value_domain != domain or self._value_resolution != resolution:
            self._readValue(domain, resolution)

            # Correct negative amplitude solutions by taking the absolute value
            # of the amplitude and rotating the phase by 180 deg.
            if self._isPolar and (self._calType == 'Gain' or self._calType == 'DirectionalGain'):
                self._value[1][self._value[0] < 0.0] += numpy.pi
                self._value[0] = numpy.abs(self._value[0])

    def _readValue(self, domain=None, resolution=None):
        if self._elements is None:
            value = numpy.array(self.__fetch_value(self._name, domain, resolution))
            self._value = (value, value)
        else:
            el0 = None
            if not self._elements[0] is None:
                el0 = numpy.array(self.__fetch_value(self._elements[0], domain, resolution))

            el1 = None
            if not self._elements[1] is None:
                el1 = numpy.array(self.__fetch_value(self._elements[1], domain, resolution))

            assert((not el0 is None) or (not el1 is None))

            if el0 is None:
                el0 = numpy.zeros(el1.shape)

            if el1 is None:
                el1 = numpy.zeros(el0.shape)

            self._value = [el0, el1]

        self._value_domain = domain
        self._value_resolution = resolution

    def __fetch_value(self, name, domain=None, resolution=None):
        if domain is None:
            tmp = self._db.getValuesGrid(name)[name]
        else:
            if resolution is None:
                tmp = self._db.getValuesGrid(name, domain[0], domain[1], domain[2], domain[3])[name]
            else:
                tmp = self._db.getValuesStep(name, domain[0], domain[1], resolution[0], domain[2], domain[3], resolution[1])[name]
        
        # store frequencies
        self._freq = numpy.array([tmp["freqs"]] * len(tmp["values"]))

        # store times and frequencies
        self._freqs = numpy.array(tmp["freqs"])
        self._times = numpy.array(tmp["times"])
        return tmp["values"]

class PlotWindow(QFrame):
    def __init__(self, parms, selection, resolution=None, parent=None, title=None):
        QFrame.__init__(self, parent)

        if not title is None:
            self.setWindowTitle(title)

        self.parms = parms
        self.selected_parms = [self.parms[i] for i in selection]

        self.calType = self.selected_parms[0]._calType

        self.fig = Figure((5, 4), dpi=100)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)

        self.axis = 0
        self.index = 0
        self.zoom = 0
        self.valminmax = [[],[]]
        self.blocked_valminmax = [[],[]]

        self.xminmax =[]

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.axisSelector = QComboBox()
        self.axisSelector.addItem("Frequency")
        self.axisSelector.addItem("Time")
        self.connect(self.axisSelector, SIGNAL('activated(int)'), self.handle_axis)

        self.spinner = QSpinBox()
        self.connect(self.spinner, SIGNAL('valueChanged(int)'), self.handle_spinner)

        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setTracking(False)
        self.slider.setToolTip('Zoom in on the median of amplitude')
        self.connect(self.slider, SIGNAL('valueChanged(int)'), self.handle_slider)

        self.show_legend = False
        self.legendCheck = QCheckBox("Legend")
        self.connect(self.legendCheck, SIGNAL('stateChanged(int)'), self.handle_legend)

        self.block_axis = False
        self.blockaxisCheck = QCheckBox("Block y-axis")
        self.blockaxisCheck.setToolTip('Block y-axis when stepping through time or frequency')
        self.connect(self.blockaxisCheck, SIGNAL('stateChanged(int)'), self.handle_blockaxis)

        self.use_points = False
        self.usepointsCheck = QCheckBox("Use points")
        self.usepointsCheck.setToolTip('Use points for plotting amplitude')
        self.connect(self.usepointsCheck, SIGNAL('stateChanged(int)'), self.handle_usepoints)

        self.valuesonxaxis = False
        self.valuesonxaxisCheck = QCheckBox("Values on x-axis")
        self.valuesonxaxisCheck.setChecked(False)
        self.valuesonxaxisCheck.setToolTip('Plot the value of time / frequency on x-axis (if unchecked, the sample number is shown)')
        self.connect(self.valuesonxaxisCheck, SIGNAL('stateChanged(int)'), self.handle_valuesonxaxis)

        self.polar = True
        self.polarCheck = QCheckBox("Polar")
        self.polarCheck.setChecked(True)
        self.connect(self.polarCheck, SIGNAL('stateChanged(int)'), self.handle_polar)

        self.unwrap_phase = False
        self.unwrapCheck = QCheckBox("Unwrap phase")
        self.connect(self.unwrapCheck, SIGNAL('stateChanged(int)'), self.handle_unwrap)

        self.sum_parm = []
        self.sum_index = []
        self.sumSelector = QComboBox()
        self.sumSelector.setToolTip('Show the sum of multiple phase effects (\'+\': include in sum)')
        self.sumSelector.addItem('+'+self.selected_parms[0]._name)
        # select only parm with the same antenna, exclude *RotationAngle: that's no phase
        self.idxParmsSameAnt = [i for i, parm in enumerate(self.parms) if parm._antenna in \
                [parm2._antenna for parm2 in self.selected_parms] \
                and parm._calType!="CommonRotationAngle" and parm._calType!="RotationAngle"
                and parm._name!=self.selected_parms[0]._name]
        for idx in self.idxParmsSameAnt:
            self.sumSelector.addItem(parms[idx]._name)
        self.connect(self.sumSelector, SIGNAL('activated(int)'), self.handle_sum)
        self.sumSelector.setEnabled(False)

        self.sumLabel = QLabel("Phase sum:")
        self.sumLabel.setEnabled(False)

        self.reference_parm = None
        self.reference_index = 0
        self.referenceSelector = QComboBox()
        self.referenceSelector.addItem("None")
        self.idxParmsSameType = [i for i, parm in enumerate(self.parms) if parm._calType == self.calType]
        for idx in self.idxParmsSameType:
            self.referenceSelector.addItem(parms[idx]._name)
        self.connect(self.referenceSelector, SIGNAL('activated(int)'), self.handle_reference)

        self.referenceLabel = QLabel("Phase reference:")

        self.domain = common_domain(self.selected_parms)

        self.resolution = None
        if (not self.domain is None) and (not resolution is None):
            self.resolution = [min(max(resolution[0], 1.0), self.domain[1] - self.domain[0]),
                min(max(resolution[1], 1.0), self.domain[3] - self.domain[2])]

        self.shape = (1, 1)
        if not self.domain is None:
            self.shape = (self.selected_parms[0].valueAmp(self.domain, self.resolution).shape)
            assert(len(self.shape) == 2)

        self.spinner.setRange(0, self.shape[1 - self.axis] - 1)

        # display widgets
        hbox = QHBoxLayout()
        hbox.addWidget(self.axisSelector)
        hbox.addWidget(self.spinner)
        hbox.addWidget(self.legendCheck)

        # Re/Imag or Ampl/Phase only relevant for gains
        if self.calType=="Gain" or self.calType=="DirectionalGain":
            hbox.addWidget(self.polarCheck)
        hbox.addWidget(self.unwrapCheck)
        hbox.addWidget(self.blockaxisCheck)

        # Phases are always points, amplitudes (gains) are lines by default
        if self.calType=="Gain" or self.calType=="DirectionalGain":
            hbox.addWidget(self.usepointsCheck)
        hbox.addWidget(self.valuesonxaxisCheck)
        hbox.addStretch(1)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.referenceLabel)
        hbox2.addWidget(self.referenceSelector)
        # For now, phase sum only works when only one parameter is shown
        if len(self.selected_parms)==1 and \
           (self.calType!="CommonRotationAngle" and 
            self.calType!="RotationAngle"):
            hbox2.addWidget(self.sumLabel)
            hbox2.addWidget(self.sumSelector)
        hbox2.addStretch(1)
        hbox3 = QHBoxLayout()
   
        # Zooming in only possible on amplitudes (gains)
        if self.calType=="Gain" or self.calType=="DirectionalGain":
            hbox3.addWidget(self.slider)
        hbox3.addWidget(self.canvas)
        layout = QVBoxLayout()
        layout.addLayout(hbox)
        
        # RotationAngles do not have phase sum 
        layout.addLayout(hbox2)
        layout.addLayout(hbox3)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

        self.plot()

    def handle_spinner(self, index):
        self.index = index
        self.plot()

    def handle_slider(self, zoom):
        self.zoom = zoom
        self.resize_plot()
        self.canvas.draw()

    def handle_axis(self, axis):
        if axis != self.axis:
            self.axis = axis
            self.spinner.setRange(0, self.shape[1 - self.axis] - 1)
            self.spinner.setValue(0)
            self.plot()

    def find_reference(self, parm, ant_parm):
        """
        find the parm which is a suitable reference for "parm".
        The reference antenna is taken from ant_parm
        """
        reference_antenna = ant_parm._antenna
        reference_name = parm._name.replace(parm._antenna, reference_antenna)
        reference_parm = next((p for p in self.parms if p._name == reference_name), None)
        if reference_parm == None:
            print "ERROR: cannot find a suitable reference (", reference_name ,") for", parm._name
            return  None
        else:
#            print "DEBUG: using reference ", reference_name ," for", parm._name
            return reference_parm

    def handle_sum(self, index):
        if index == 0:
            a=3 #do nothing
            # self.sum_parm = []
            # remove all the "+" from the drop down menu
            # for sum_index in self.sum_index:
            #    parm = self.parms[self.idxParmsSameAnt[sum_index-1]]
            #    self.sumSelector.setItemText(sum_index, parm._name)
            # self.sum_index = []
        elif index in self.sum_index:
            parm = self.parms[self.idxParmsSameAnt[index-1]]
            self.sum_parm.pop(self.sum_index.index(index))
            self.sum_index.remove(index)
            # remove "+" from the selected element
            self.sumSelector.setItemText(index, parm._name)
        else:
            parm = self.parms[self.idxParmsSameAnt[index-1]]
            ant_parm = self.parms[self.idxParmsSameType[self.reference_index-1]]
            sum_reference_parm = self.find_reference(parm, ant_parm)
            # add a tuple of the parm to sum and its reference
            self.sum_parm.append( (parm, sum_reference_parm) )
            self.sum_index.append(index)
            # add "+" to the selected element
            self.sumSelector.setItemText(index, '+'+parm._name)
        self.plot()

    def handle_legend(self, state):
        self.show_legend = (state == 2)
        self.plot()

    def handle_unwrap(self, state):
        self.unwrap_phase = (state == 2)
        self.plot()

    def handle_blockaxis(self, state):
        self.block_axis = (state == 2)
        if self.block_axis:
            self.slider.setEnabled(False)
            for i, ax in enumerate(self.fig.axes):
                self.blocked_valminmax[i] = (ax.axis()[2], ax.axis()[3], 0)
        else:
            self.slider.setEnabled(True)
            self.plot()

    def handle_usepoints(self, state):
        self.use_points = (state == 2)
        self.plot()

    def handle_valuesonxaxis(self, state):
        self.valuesonxaxis = (state == 2)
        self.plot()

    def handle_polar(self, state):
        self.polar = (state == 2)
        self.referenceSelector.setEnabled(self.polar)
        self.referenceLabel.setEnabled(self.polar)
        self.sumSelector.setEnabled(self.polar)
        self.sumLabel.setEnabled(self.polar)
        self.unwrapCheck.setEnabled(self.polar)
        self.plot()

    def handle_reference(self, index):
        if index != self.reference_index:
            if index == 0:
                self.reference_parm = None
                self.sum_parm = []
                # remove all the "+" from the drop down menu
                for sum_index in self.sum_index:
                    parm = self.parms[self.idxParmsSameAnt[sum_index-1]]
                    self.sumSelector.setItemText(sum_index, parm._name)
                self.sum_index = []
            else:
                self.reference_parm = self.parms[self.idxParmsSameType[index-1]]

                # update all the sums to use the new reference
                for i, sum_parm in enumerate(self.sum_parm):
                    sum_reference_parm = self.find_reference(sum_parm[0], self.reference_parm)
                    # add a tuple of the parm to sum and its reference
                    self.sum_parm[i] = (sum_parm[0], sum_reference_parm)

            self.sumSelector.setEnabled((index != 0))
            self.sumLabel.setEnabled((index != 0))
            self.reference_index = index
            self.plot()

    def resize_plot(self):
        """
        Set y-axis scale looking the slider and the number of samples.
        """
        for i, ax in enumerate(self.fig.axes):
            # handle single x value (typically one freq)
            if self.shape[self.axis] != 1:
                #ax.set_xlim(0, self.shape[self.axis] - 1)
                ax.set_xlim(self.xminmax[0],self.xminmax[1])
            if self.block_axis == True:
                ax.set_ylim(self.blocked_valminmax[i][0], self.blocked_valminmax[i][1])
            else:
                # handle single y value (typically y=0)
                if self.valminmax[i][0] == self.valminmax[i][1]:
                    ax.set_ylim(self.valminmax[i][0]-1e-5, self.valminmax[i][0]+1e-5)
                elif self.zoom == 0 or (self.polar and i>0):
                    ax.set_ylim(self.valminmax[i][0], self.valminmax[i][1])
                else:
                    rng = (self.valminmax[i][1] - self.valminmax[i][0])/numpy.exp(self.zoom/10.)
                    ax.set_ylim(ymin = self.valminmax[i][2] - rng/2., \
                                ymax = self.valminmax[i][2] + rng/2.)

    def plot(self):
        """
        Prepare the plotting space, double makes a double plot (only [Directional]Gain)
        """

        phase = []
        amp = []
        labels = []
        xvalues = []

        # Selector for double plot
        if self.calType == 'Gain' or self.calType == 'DirectionalGain':
            plot_type = 'double'
        elif self.calType == 'CommonScalarAmplitude':
            plot_type = 'amp'
        else:
            self.usepointsCheck.setEnabled(False)
            plot_type = 'ph'

        if not self.domain is None:
            for parm in self.selected_parms:

                if plot_type == 'double' or plot_type == 'ph':
                    valuePhase = parm.valuePhase(self.domain, self.resolution, asPolar = self.polar, \
                        unwrap_phase=self.unwrap_phase, reference_parm=self.reference_parm, sum_parms=self.sum_parm)

                    if self.axis == 0: # time on x-axis
                        phase.append(valuePhase[:, self.index])
                    else:              # freq on x-axis
                        phase.append(valuePhase[self.index, :])

                if plot_type == 'double' or plot_type == 'amp':
                    valueAmp = parm.valueAmp(self.domain, self.resolution, asPolar = self.polar)

                    if self.axis == 0: # time on x-axis
                        amp.append(valueAmp[:, self.index])
                    else:              # freq on x-axis
                        amp.append(valueAmp[self.index, :])

                if self.axis == 0: # time on x-axis 
                    xvalues.append((parm._times-parm._times[0])/60.) 
                else:              # freq on x-axis
                    xvalues.append(parm._freqs/1.e6)

                if not self.valuesonxaxis:
                    xvalues[-1] = range(len(xvalues[-1]))
    
                self.xminmax=[xvalues[0][0],xvalues[0][-1]]
               
                labels.append(parm._name)

        legend = self.show_legend and len(labels) > 0
        if self.valuesonxaxis:
            xlabel = ["Time (minutes since start)", "Freq (MHz)"][self.axis]
        else:
            xlabel = ["Time (sample)", "Freq (sample)"][self.axis]

        if self.calType == "CommonRotationAngle" or self.calType == "RotationAngle" or self.calType == "RotationMeasure":
            phaselabel = "Rotation angle (rad)"
        else:
            phaselabel = "Phase (rad)"


        if plot_type == 'double':
            # put nans to 0
            [numpy.putmask(amp[i], amp[i]!=amp[i], 0) for i in xrange(len(amp))]
            [numpy.putmask(phase[i], phase[i]!=phase[i], 0) for i in xrange(len(phase))]
            if self.polar:
                    self.valminmax[0] = plot(self.fig, amp, x=xvalues, sub="211", labels=labels, show_legend=legend, xlabel=xlabel, ylabel="Amplitude", scatter=self.use_points)
                    self.valminmax[1] = plot(self.fig, phase, x=xvalues, clf=False, sub="212", stack=True, scatter=True, labels=labels, show_legend=legend, xlabel=xlabel, ylabel=phaselabel)
            else:
                    self.valminmax[0] = plot(self.fig, amp, x=xvalues, sub="211", labels=labels, show_legend=legend, xlabel=xlabel, ylabel="Real", scatter=self.use_points)
                    self.valminmax[1] = plot(self.fig, phase, x=xvalues, clf=False, sub="212", labels=labels, show_legend=legend, xlabel=xlabel, ylabel="Imaginary", scatter=self.use_points)
        elif plot_type == 'ph':
            # put nans to 0
            [numpy.putmask(phase[i], phase[i]!=phase[i], 0) for i in xrange(len(phase))]
            self.valminmax[0] = plot(self.fig, phase, x=xvalues, sub="111", stack=True, scatter=True, labels=labels, show_legend=legend, xlabel=xlabel, ylabel=phaselabel)
        elif plot_type == 'amp':
            # put nans to 0
            [numpy.putmask(amp[i], amp[i]!=amp[i], 0) for i in xrange(len(amp))]
            self.valminmax[0] = plot(self.fig, amp, x=xvalues, sub="111", labels=labels, show_legend=legend, xlabel=xlabel, ylabel="Amplitude", scatter=self.use_points)

        self.resize_plot()
        self.canvas.draw()


class DefValuesWindow(QFrame):
    def __init__(self, db, parent=None, title=None):
        QFrame.__init__(self,None)

        defvalues=db.getDefValues()

        layout = QVBoxLayout()
        self.table = QTableWidget(len(defvalues),2)

        row=0
        for defname in defvalues:
          self.table.setItem(row, 0, QTableWidgetItem(defname))
          valItem = QTableWidgetItem(str(defvalues[defname][0][0]))
          valItem.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
          self.table.setItem(row, 1, QTableWidgetItem(valItem))
          row = row+1


        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.horizontalHeader().setResizeMode(QHeaderView.Fixed)
        self.table.horizontalHeader().setResizeMode(1,QHeaderView.Stretch)
        self.table.resizeColumnsToContents();
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self.table,1)
        self.setWindowTitle(title)

        self.db=db

        self.setLayout(layout);


class MainWindow(QFrame):
    def __init__(self, db, windowname):
        QFrame.__init__(self)
        self.setWindowTitle(windowname)
        self.db = db
        self.figures = []
        self.parms = []
        self.windowname=windowname

        layout = QVBoxLayout()

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.list, 1)

        self.useResolution = False
        checkResolution = QCheckBox("Use resolution")
        checkResolution.setChecked(False)
        self.connect(checkResolution, SIGNAL('stateChanged(int)'), self.handle_resolution)

        self.resolution = [QLineEdit(), QLineEdit()]
        self.resolution[0].setAlignment(Qt.AlignRight)
        self.resolution[1].setAlignment(Qt.AlignRight)

        hbox = QHBoxLayout()
        hbox.addWidget(checkResolution)
        hbox.addWidget(self.resolution[0])
        hbox.addWidget(QLabel("Hz"))
        hbox.addWidget(self.resolution[1])
        hbox.addWidget(QLabel("s"))
        layout.addLayout(hbox)

        self.plot_button = QPushButton("Plot")
        self.connect(self.plot_button, SIGNAL('clicked()'), self.handle_plot)
        self.defvalues_button = QPushButton("DefValues")
        self.connect(self.defvalues_button, SIGNAL('clicked()'), self.handle_defvalues)
        self.close_button = QPushButton("Close figures")
        self.connect(self.close_button, SIGNAL('clicked()'), self.handle_close)

        hbox = QHBoxLayout()
        hbox.addWidget(self.plot_button)
        hbox.addWidget(self.defvalues_button)
        hbox.addWidget(self.close_button)
        layout.addLayout(hbox)

        self.setLayout(layout)
        self.populate()

    def populate(self):
        names = self.db.getNames()
        while len(names) > 0:
            parm = names.pop()
            split = parm.split(":")

            # just one entry for Real/Imag or Amp/Ph
            if contains(split, "Real") or contains(split, "Imag"):
                if contains(split, "Real"):
                    idx = split.index("Real")
                    split[idx] = "Imag"
                    other = ":".join(split)
                    try:
                        names.pop(names.index(other))
                    except ValueError:
                        other = None
                    elements = [parm, other]
                else:
                    idx = split.index("Imag")
                    split[idx] = "Real"
                    other = ":".join(split)
                    try:
                        names.pop(names.index(other))
                    except ValueError:
                        other = None
                    elements = [other, parm]

                split.pop(idx)
                self.parms.append(Parm(self.db, ":".join(split), elements, isPolar = False))

            elif contains(split, "Ampl") or contains(split, "Phase"):
                if contains(split, "Ampl"):
                    idx = split.index("Ampl")
                    split[idx] = "Phase"
                    other = ":".join(split)
                    try:
                        names.pop(names.index(other))
                    except ValueError:
                        other = None
                    elements = [parm, other]
                else:
                    idx = split.index("Phase")
                    split[idx] = "Ampl"
                    other = ":".join(split)
                    try:
                        names.pop(names.index(other))
                    except ValueError:
                        other = None
                    elements = [other, parm]

                split.pop(idx)
                self.parms.append(Parm(self.db, ":".join(split), elements, isPolar = True))
            else:
                self.parms.append(Parm(self.db, parm))

        self.parms = [parm for parm in self.parms if not parm.empty()]
        self.parms.sort(cmp=lambda x, y: cmp(x._name, y._name))

        domain = common_domain(self.parms)
        if not domain is None:
            self.resolution[0].setText("%.3f" % ((domain[1] - domain[0]) / 100.0))
            self.resolution[1].setText("%.3f" % ((domain[3] - domain[2]) / 100.0))

        for parm in self.parms:
            name = parm._name
            if parm._isPolar and (parm._calType == 'Gain' or parm._calType == 'DirectionalGain'):
                name = "%s (polar)" % name

            it = QListWidgetItem(name, self.list)
            it.setBackground(parm.color())

    def close_all_figures(self):
        for figure in self.figures:
            figure.close()

        self.figures = []

    def handle_resolution(self, state):
        self.useResolution = (state == 2)

    def handle_plot(self):
        selection = [self.list.row(item) for item in self.list.selectedItems()]
        selection.sort()

        resolution = None
        if self.useResolution:
            resolution = [float(item.text()) for item in self.resolution]

        # plot together only parms of the same type
        calTypes = set([self.parms[idx]._calType for idx in selection])
        for calType in calTypes:
            this_selection = [idx for idx in selection if self.parms[idx]._calType == calType]

            self.figures.append(PlotWindow(self.parms, this_selection, resolution, title=self.windowname + ": Figure %d" % (len(self.figures) + 1)))
            
            self.figures[-1].show()

    def handle_defvalues(self):
        self.defvalueswindow=DefValuesWindow(db, title=parmdbname+': default Values')
        self.defvalueswindow.show()

    def handle_close(self):
        self.close_all_figures()

    def closeEvent(self, event):
        self.close_all_figures()
        event.accept()


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] == "--help":
        print "usage: parmdbplot.py <parmdb>"
        sys.exit(1)

    try:
        db = parmdb.parmdb(sys.argv[1])
    except:
        print "ERROR:", sys.argv[1], "is not a valid parmdb."
        sys.exit(1)

    app = QApplication(sys.argv)

    # show parmdbname in title (with msname if it is inside an MS)
    splitpath = string.split(string.rstrip(sys.argv[1],"/"),"/")
    if len(splitpath)>1 and splitpath[-2][-2:]=="MS":
      parmdbname = "/".join(splitpath[-2:])
    else:
      parmdbname = splitpath[-1]

    window = MainWindow(db, parmdbname)
    window.show()

#    app.connect(app, SIGNAL('lastWindowClosed()'), app, SLOT('quit()'))
    app.exec_()
