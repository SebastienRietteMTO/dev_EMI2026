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

from microphy import microphyFactory
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
sedim_SPLI2 = pppy_sedimentation_PHYEX(dt=60., method='step-by-step',
                                       name="Eulerian 2M scheme with dt=60.s",
                                       tag="SPLI2_dt=60.",
                                       namel=json.dumps({'NAM_PARAM_LIMA': {'LSEDC':True,
                                                                            'LSEDI':True,
                                                                            'NMOM_C':2,
                                                                            'NMOM_R':2,
                                                                            'NMOM_I':2,
                                                                            'NMOM_S':2,
                                                                            'NMOM_G':2,
                                                                            'NMOM_H':2},
                                                         'NAM_NEBn': {'LCONDBORN': True}}))
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

# Compute initial concentration using ICE3 formulations
XBOLTZ = 1.380658E-23
XAVOGADRO = 6.0221367E+23
XMD = 28.9644E-3
XMV    = 18.0153E-3
XRD = XAVOGADRO * XBOLTZ / XMD
XRV    = XAVOGADRO * XBOLTZ / XMV
XCPD = 7.* XRD / 2.
exner = (init_state['P'] / 1.E5) ** (XRD / XCPD)
rhod = init_state['P'] / ((XRD + init_state['rv'] * XRV) * init_state['Theta'] * exner)
for spec in ('c', 'r', 'i', 's', 'g'):
    # Concentration are in #/kg of dry air but ICE3 uses #/m3
    if spec == 'c':
        init_state[f"n{spec}"] = op_profile.copy() * 300.E6 / rhod
    elif spec == 'i':
        rhoi = 900
        vmi = rhod * init_state['ri']
        N = vmi * numpy.maximum(0.05E6, -0.15319E6 - 0.021454E6 * numpy.log(vmi)) ** 3 / \
            (4 * numpy.pi * rhoi)
        N[vmi == 0.] = 0.
        init_state[f"n{spec}"] = N / rhod
    else:
        mp = microphyFactory(f'ICE:{spec}')
        init_state[f"n{spec}"] = mp.number_concentration(init_state[f'r{spec}'] * rhod) / rhod

#Comparison of different schemes
conf = {'schemes': [sedim_STAT, sedim_SPLI, sedim_SPLI2, sedim_SREF],
        'output_dir': output_dir,
        'duration': 3600.,
        'init_state': init_state,
        'name': "First sedimentation test",
        'tag': "firstSedimTest"}
comp = PPPYComp(**conf)
comp.run(force=False)
figs = []

#rr for the different schemes
levels = [1.E-4, 1.E-3, 5.E-3, 1.E-2, 2.E-2, 3.E-2, 4.E-2, 5.E-2, 6.E-2, 7.E-2, 8.E-2, 9.E-2]
confplt = {'var_names': ['rr'], 'y_var': 'Z_mass', 'mfactor': 1000., 'contourlevels': levels}
plot1 = 'evol', {'only_param_tag': [sedim_STAT.tag], **confplt}
plot2 = 'evol', {'only_param_tag': [sedim_SPLI.tag], **confplt}
plot3 = 'evol', {'only_param_tag': [sedim_SPLI2.tag], **confplt}
plot4 = 'evol', {'only_param_tag': [sedim_SREF.tag], **confplt}
fig, plots = comp.plot_multi((4, 1), [plot1, plot2, plot3, plot4])
figs.append(fig)

#nr for LIMA
levels = [0, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000]
confplt = {'var_names': ['nr'], 'y_var': 'Z_mass', 'contourlevels': levels}
plot = 'evol', {'only_param_tag': [sedim_SPLI2.tag], **confplt}
fig, plots = comp.plot_multi((1, 1), [plot])
figs.append(fig)

#cum_r for the different schemes
plot = 'evol', {'var_names': ['cum_r']}
fig, plots = comp.plot_multi((1, 1), [plot])
figs.append(fig)

plt.show()
for fig in figs:
    plt.close(fig)
