# -*- coding: utf-8 -*-
"""
Gamut Boundary Descriptor (GDB) - Morovic and Luo (2000)
========================================================

Defines the * Morovic and Luo (2000)* *Gamut Boundary Descriptor (GDB)*
computation objects:

-   :func:`colour.gamut.gamut_boundary_descriptor_Morovic2000`

See Also
--------
`Gamut Boundary Descriptor Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/gamut/boundary.ipynb>`_

References
----------
-   :cite:`` :
"""

from __future__ import division, unicode_literals

import numpy as np
from colour.algebra import cartesian_to_polar
from colour.utilities import (as_int_array, as_float_array, linear_conversion,
                              tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['gamut_boundary_descriptor_Morovic2000']


def gamut_boundary_descriptor_Morovic2000(Jab,
                                          E=np.array([50, 0, 0]),
                                          m=16,
                                          n=16):
    Jab = as_float_array(Jab).reshape(-1, 3)
    E = as_float_array(E)

    phi, a, b = tsplit(Jab - E)
    r, alpha = tsplit(cartesian_to_polar(tstack([a, b])))

    GDB_m = np.full([m, n, 3], np.nan)

    # Lightness :math:`J` is in range [-E_{J}, E_{J}], :math:`\\phi` indexes
    # are in range [0, m - 1].
    phi_i = np.floor(linear_conversion(phi, (-E[0], E[0]), (0, m)))
    phi_i = as_int_array(np.clip(phi_i, 0, m - 1))

    # Polar coordinates are in range [-pi, pi], :math:`\\alpha` indexes are
    # in range [0, n - 1].
    alpha_i = np.floor(linear_conversion(alpha, (-np.pi, np.pi), (0, n)))
    alpha_i = as_int_array(np.clip(alpha_i, 0, n - 1))

    for i in range(m):
        for j in range(n):
            i_j = np.intersect1d(
                np.argwhere(phi_i == i), np.argwhere(alpha_i == j))

            if i_j.size == 0:
                continue

            r_m = np.argmax(r[i_j]).tolist()

            GDB_m[i, j] = Jab[i_j[r_m]]

    # Naive non-vectorised implementation kept for reference.
    # :math:`r_m` is used to keep track of the maximum :math:`r` value.
    # r_m = np.full([m, n, 1], np.nan)
    # for i, Jab_i in enumerate(Jab):
    #     a_i, p_i = alpha_i[i], phi_i[i]
    #     r_i_j = r_m[a_i, p_i]
    #
    #     if r[i] > r_i_j or np.isnan(r_i_j):
    #         GDB_m[a_i, p_i] = Jab_i
    #         r_m[a_i, p_i] = r[i]

    return GDB_m
