#!/usr/bin/env python3

# Copyright (c) Météo France (2018-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info

"""
This modules is an example of comparison for the sedimentation schemes
"""

import os
import json
import numpy
import matplotlib.pyplot as plt
from pathlib import Path

from pppy import PPPYComp
from pppy_sedimentation_PHYEX import pppy_sedimentation_PHYEX

#Configuration of comparison
output_dir = os.path.dirname(os.path.abspath(__file__))

#Configuration of schemes
sedim_STAT = pppy_sedimentation_PHYEX(dt=60., method='step-by-step',
                                      name="Statistical scheme with dt=60.s",
                                      tag="STAT_dt=60.",
                                      namel=json.dumps({'NAM_PARAM_ICEn': {'LSEDIC':True,
                                                                           'CSEDIM':'STAT'}}))
sedim_SPLI = pppy_sedimentation_PHYEX(dt=60., method='step-by-step',
                                      name="Eulerian scheme with dt=60.s",
                                      tag="SPLI_dt=60.",
                                      namel=json.dumps({'NAM_PARAM_ICEn': {'LSEDIC':True,
                                                                           'CSEDIM':'SPLI',
                                                                           'XSPLIT_MAXCFL':.8}}))
sedim_SREF = pppy_sedimentation_PHYEX(dt=60., method='one-step',
                                      name="Reference scheme (eulerian scheme run " + \
                                           "in one-step mode) with dt=60.s",
                                      tag="SREF_dt=60.",
                                      namel=json.dumps({'NAM_PARAM_ICEn': {'LSEDIC':True,
                                                                           'CSEDIM':'SPLI',
                                                                           'XSPLIT_MAXCFL':.8}}))

#Initial state
NIJT, NKT = 1, 20
op_profile = numpy.zeros((NKT, NIJT)) #vertical profile of hydrometeor
op_profile[NKT - 3, :] = 1 #Only one point with not null content
dzz = numpy.ones((NKT, NIJT)) * 50 #distance between two flux levels
Z_flux = dzz.cumsum(axis=0) - dzz[0,:]
Z_mass = numpy.ndarray((NKT, NIJT))
Z_mass[:-1, :] = .5*(Z_flux[1:, :]+Z_flux[:-1, :])
Z_mass[-1, :] = 2 * Z_mass[-2, :] - Z_mass[-3, :]

init_state = {'P': numpy.ones((NKT, NIJT)) * 101325.,
              'Theta': numpy.ones((NKT, NIJT)) * 280.,
              'rv': numpy.ones((NKT, NIJT)) * 0.001,
              'rc': op_profile.copy() * 1.E-4,
              'rr': op_profile.copy() * 1.E-4,
              'ri': op_profile.copy() * 1.E-4,
              'rs': op_profile.copy() * 1.E-4,
              'rg': op_profile.copy() * 1.E-4,
              'dzz': dzz,
              'Z_mass': Z_mass,
              'sea': numpy.zeros((NIJT, )),
              'town': numpy.zeros((NIJT, )),
              'cum_c': numpy.zeros((NIJT, )),
              'cum_r': numpy.zeros((NIJT, )),
              'cum_s': numpy.zeros((NIJT, )),
              'cum_g': numpy.zeros((NIJT, ))}

#Comparison of different schemes
conf = {'schemes': [sedim_STAT, sedim_SPLI, sedim_SREF],
        'output_dir': output_dir,
        'duration': 3600.,
        'init_state': init_state,
        'name': "First sedimentation test",
        'tag': "firstSedimTest"}
comp = PPPYComp(**conf)
comp.run(force=False)
figs = []
levels = [1.E-4, 1.E-3, 5.E-3, 1.E-2, 2.E-2, 3.E-2, 4.E-2, 5.E-2, 6.E-2, 7.E-2, 8.E-2, 9.E-2]

#rr for the different schemes
confplt = {'var_names': ['rr'], 'y_var': 'Z_mass', 'mfactor': 1000., 'contourlevels': levels}
plot1 = 'evol', {'only_param_tag': [sedim_STAT.tag], **confplt}
plot2 = 'evol', {'only_param_tag': [sedim_SPLI.tag], **confplt}
plot3 = 'evol', {'only_param_tag': [sedim_SREF.tag], **confplt}
fig, plots = comp.plot_multi((3, 1), [plot1, plot2, plot3])
figs.append(fig)

plt.savefig(Path('../fig/figure1'))

#cum_r for the different schemes
plot = 'evol', {'var_names': ['cum_r']}
fig, plots = comp.plot_multi((1, 1), [plot])
figs.append(fig)

plt.savefig(Path('../fig/figure2'))

plt.show()
for fig in figs:
    plt.close(fig)
