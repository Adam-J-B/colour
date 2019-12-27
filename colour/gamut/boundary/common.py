# -*- coding: utf-8 -*-
"""
Common Gamut Boundary Descriptor (GDB) Utilities
================================================

Defines various *Gamut Boundary Descriptor (GDB)* common utilities.

-   :func:`colour.interpolate_gamut_boundary_descriptor`

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
import scipy.interpolate
import scipy.ndimage
from colour.utilities import orient, tstack, warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'interpolate_gamut_boundary_descriptor'
]
def interpolate_gamut_boundary_descriptor(GDB_m):
    GDB_m = np.asarray(GDB_m)

    r_slice = np.s_[:GDB_m.shape[0], ...]
    # If bounding columns have NaN, :math:`GDB_m` matrix is tiled
    # horizontally so that right values interpolate with left values and
    # vice-versa.
    if np.any(np.isnan(GDB_m[..., 0])) or np.any(np.isnan(GDB_m[..., -1])):
        warning(
            'Gamut boundary descriptor matrix bounding columns contains NaN '
            'and will be horizontally tiled!')
        GDB_m_i = np.hstack([GDB_m] * 3)
        r_slice = np.s_[GDB_m.shape[0]:GDB_m.shape[0] * 2]

    c_slice = np.s_[:GDB_m.shape[1], ...]
    # If bounding rows have NaN, :math:`GDB_m` matrix is reflected vertically
    # so that top and bottom values are replicated via interpolation, i.e.
    # equivalent to nearest-neighbour interpolation.
    if np.any(np.isnan(GDB_m[0, ...])) or np.any(np.isnan(GDB_m[-1, ...])):
        warning('Gamut boundary descriptor matrix bounding rows contains NaN '
                'and will be vertically reflected!')
        GDB_m_f = orient(GDB_m_i, 'Flop')
        GDB_m_i = np.vstack([GDB_m_f, GDB_m_i, GDB_m_f])
        c_slice = np.s_[GDB_m.shape[1]:GDB_m.shape[1] * 2]

    mask = np.any(~np.isnan(GDB_m_i), axis=-1)
    for i in range(3):
        x = np.linspace(0, 1, GDB_m_i.shape[0])
        y = np.linspace(0, 1, GDB_m_i.shape[1])
        x_g, y_g = np.meshgrid(x, y, indexing='ij')
        values = GDB_m_i[mask]

        GDB_m_i[..., i] = scipy.interpolate.griddata(
            (x_g[mask], y_g[mask]),
            values[..., i], (x_g, y_g),
            method='linear')

    return GDB_m_i[r_slice, c_slice, :]


if __name__ == '__main__':
    import colour
    from colour.gamut import gamut_boundary_descriptor_Morovic2000
    
    np.set_printoptions(
        formatter={'float': '{:0.1f}'.format}, linewidth=2048, suppress=True)

    t = 3
    Hab = np.tile(np.arange(0, 360 + 45, 45) / 360, t)
    C = np.ones(len(Hab))
    # C = np.hstack([
    #     np.ones(int(len(Hab) / t)) * 1.0,
    #     np.ones(int(len(Hab) / t)) * 0.5,
    #     np.ones(int(len(Hab) / t)) * 0.25
    # ])
    L = np.hstack([
        np.ones(int(len(Hab) / t)) * 1.0,
        np.ones(int(len(Hab) / t)) * 0.5,
        np.ones(int(len(Hab) / t)) * 0.25
    ])

    LCHab = tstack([L, C, Hab])
    Jab = colour.convert(
        LCHab, 'CIE LCHab', 'CIE Lab', verbose={'describe': 'short'}) * 100

    np.random.seed(8)
    RGB = np.random.random([25, 25, 3])
    Jab_E = colour.convert(
        RGB, 'RGB', 'CIE Lab', verbose={'describe': 'short'}) * 100

    # Jab = Jab_E
    # print(Jab)

    GDB = gamut_boundary_descriptor_Morovic2000(Jab, [50, 0, 0], 5, 9)
    # print(GDB)

    # a = np.full([5, 9], np.nan)
    #
    # a[0, 0] = 1
    # a[0, -1] = 5
    # a[-1, 0] = 10
    # a[-1, -1] = 20
    #
    # a = np.full([5, 9], np.nan)
    #
    # a[0, 4] = 1
    # a[2, 4] = 3

    print(interpolate_gamut_boundary_descriptor(GDB)[..., 0])
    print(interpolate_gamut_boundary_descriptor(GDB)[..., 1])
    print(interpolate_gamut_boundary_descriptor(GDB)[..., 2])

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(
    #     Jab[..., 1], Jab[..., 2], Jab[..., 0], color='green', marker='^')
    ax.scatter(GDB[..., 1], GDB[..., 2], GDB[..., 0], color='red')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('J')

    plt.show()
