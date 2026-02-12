#!/usr/bin/env python3

# Copyright (c) Météo France (2018-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info

"""
This module contains the PPPY implementation for calling the sedimentation schemes
"""

import os
import tempfile
import json
import logging
import numpy
import f90nml
import pppy

class pppy_sedimentation_PHYEX(pppy.PPPY):
    """
    PPPY implementation for calling sedimentation schemes
    """

    def __init__(self, dt, method, name, tag, **namel):
        """
        In addition to dt, method, name and tag parameters
        defined in the PPPY class, this parameterization
        needs the following parameters:
        namel: json string representing the namelists to use as a dict of dict
        """
        super().__init__(dt, method, name, tag, **namel)

        self._init = None
        self._close = None
        self._param = None

    def setup(self, init_state, duration):
        """
        This method opens the shared library and do the init stuff.
        """
        super().setup(init_state, duration)

        #Opening of shared lib is done here to avoid loading it at initialization
        #Because if shared lib is loaded at init, all declared schemes will have
        #their shared lib loaded simultaneously which can be a source of error.
        #This way, shared lib is loaded in a sub-process (if class instance is used
        #through a PPPYComp instance).

        try:
            from pyphyex import PYICE4_SEDIMENTATION, PYINI_PHYEX, close
        except ModuleNotFoundError:
            logging.error("The module pyphyex has not been found. " + \
                          "This module can be built with the PHYEX package " + \
                          "(available on github). Details on the compilation " + \
                          "process can be found in the PHYEX documentation.")
            raise

        self._init = PYINI_PHYEX
        self._close = close
        self._param = PYICE4_SEDIMENTATION

        #Setup
        with tempfile.NamedTemporaryFile() as namel:
            nml = f90nml.read(namel.name)
            for k, v in json.loads(self._options['namel']).items():
                nml[k] = v
            nml.write(namel.name, force=True)
            self._init('AROME', 33, namel.name, False, 20, 0, 1, self._dt, 20.,
                       'ICE3', 'EDKF', 'TKEL',
                       LDCHANGEMODEL=True, LDDEFAULTVAL=True, LDREADNAM=True, LDCHECK=True,
                       KPRINT=0, LDINIT=True)
            os.remove('fort.20')

    def finalize(self):
        """
        We close the shared library so we are able to load a new one with the same symbol names.
        """
        super().finalize()
        self._close()

    def execute(self, previous_state, timestep, timestep_number):
        """
        This method does the actual work
        """
        super().execute(previous_state, timestep, timestep_number)
        ps = previous_state

        #Thermo
        XG = 9.80665
        XBOLTZ = 1.380658E-23
        XAVOGADRO = 6.0221367E+23
        XMD = 28.9644E-3
        XMV    = 18.0153E-3
        XRD = XAVOGADRO * XBOLTZ / XMD
        XRV    = XAVOGADRO * XBOLTZ / XMV
        XCPD = 7.* XRD / 2.
        XCPV   = 4.* XRV
        XCL    = 4.218E+3
        XCI    = 2.106E+3
        XTT    = 273.16
        XLVTT  = 2.5008E+6
        XLSTT  = 2.8345E+6

        #Dimensions
        NKT, NIJT = ps['Theta'].shape
        KRR = 6

        #Derived arrays
        exner = (ps['P'] / 1.E5) ** (XRD / XCPD)
        PRHODREF = ps['P'] / ((XRD + ps['rv'] * XRV) * ps['Theta'] * exner)
        T = ps['Theta'] * exner
        PRT = numpy.ndarray((KRR, NKT, NIJT))
        PRT[0] = ps['rv']
        PRT[1] = ps['rc']
        PRT[2] = ps['rr']
        PRT[3] = ps['ri']
        PRT[4] = ps['rs']
        PRT[5] = ps['rg']
        PRS = PRT / timestep
        PTHS = ps['Theta'] / timestep
        rhodj = ps['dzz'] * PRHODREF

        PLVFACT = (XLVTT + (XCPV - XCL) * (T - XTT)) / \
                  (XCPD + XCPV * ps['rv'] + \
                          XCL * (ps['rc'] + ps['rr']) + \
                          XCI * (ps['ri'] + ps['rs'] + ps['rg']))
        PLSFACT = (XLSTT + (XCPV - XCI) * (T - XTT)) / \
                  (XCPD + XCPV * ps['rv'] + \
                          XCL * (ps['rc'] + ps['rr']) + \
                          XCI * (ps['ri'] + ps['rs'] + ps['rg']))

        PQCT = numpy.zeros((NKT, NIJT))
        PQRT = numpy.zeros((NKT, NIJT))
        PQIT = numpy.zeros((NKT, NIJT))
        PQST = numpy.zeros((NKT, NIJT))
        PQGT = numpy.zeros((NKT, NIJT))
        PQCS = numpy.zeros((NKT, NIJT))
        PQRS = numpy.zeros((NKT, NIJT))
        PQIS = numpy.zeros((NKT, NIJT))
        PQSS = numpy.zeros((NKT, NIJT))
        PQGS = numpy.zeros((NKT, NIJT))
        PEFIELDW = numpy.zeros((NKT, NIJT))
        PCONC3D = numpy.zeros((NKT, NIJT))

        result = self._param(NIJT, NKT, 1, 0, False, False, timestep, KRR,
                             ps['dzz'], 0., PLVFACT, PLSFACT,
                             PRHODREF, ps['P'], ps['Theta'],
                             T, rhodj, PTHS, PRT, PRS, PCONC3D,
                             PQCT, PQRT, PQIT, PQST, PQGT,
                             PQCS, PQRS, PQIS, PQSS, PQGS,
                             PEFIELDW, 0, PSEA=ps['sea'],
                             PTOWN=ps['town'], missingOUT=['PFPR', 'PINPRH'])

        inst = {}
        (PTHS, PRS, inst['c'], inst['r'], inst['s'],
         inst['g'], _, _, _, _, _) = result

        ns = {}
        ns['rv'] = PRS[0] * timestep
        ns['rc'] = PRS[1] * timestep
        ns['rr'] = PRS[2] * timestep
        ns['ri'] = PRS[3] * timestep
        ns['rs'] = PRS[4] * timestep
        ns['rg'] = PRS[5] * timestep
        ns['Theta'] = PTHS * timestep

        for spe in ['c', 'r', 's', 'g']:
            if self._method == 'step-by-step':
                ns['cum_' + spe] = ps['cum_' + spe] + inst[spe] * timestep
            else:
                ns['cum_' + spe] = inst[spe] * timestep

        return ns
