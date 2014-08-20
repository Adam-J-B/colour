#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ATD (1995) Colour Vision Model
==============================

Defines *ATD (1995)* colour vision model objects:

-   :func:`ATD95_Specification`
-   :func:`XYZ_to_ATD95`

Notes
-----
-   According to *CIE TC1-34* definition of a colour appearance model, the
    *ATD95* model cannot be considered as a colour appearance model. It was
    developed with different aims and is described as a model of colour vision.

References
----------
.. [1]  **Mark D. Fairchild**, *Color Appearance Models, 3nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published June 2013, ASIN: B00DAYO8E2,
        Locations 5841-5991.
.. [2]  **S. Lee Guth**,
        *Further applications of the ATD model for color vision*,
        *IS&T/SPIE's Symposium on Electronic Imaging: Science & Technology*,
        *International Society for Optics and Photonics*,
        pages 12-26.
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'GPL V3.0 - http://www.gnu.org/licenses/'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ATD95_Specification',
           'XYZ_to_ATD95',
           'luminance_to_retinal_illuminance',
           'XYZ_to_LMS_ATD95',
           'get_opponent_colour_dimensions',
           'get_final_response']

ATD95_Specification = namedtuple('ATD95_Specification', (
    'H', 'Br', 'C', 'A_1', 'T_1', 'D_1', 'A_2', 'T_2', 'D_2'))
"""
Defines the *ATD (1995)* colour vision model specification.

Parameters
----------
H : float
    *Hue* angle :math:`H` in degrees.
Br : float
    Correlate of *brightness* :math:`Br`.
C : float
    Correlate of *saturation* :math:`C`. *Guth (1995)* incorrectly uses the
    terms saturation and chroma interchangeably. However, :math:`C` is here a
    measure of saturation rather than chroma since it is measured relative to
    the achromatic response for the stimulus rather than that of a similarly
    illuminated white.
A_1 : float
    First stage :math:`A_1` response.
T_1 : float
    First stage :math:`T_1` response.
D_1 : float
    First stage :math:`D_1` response.
A_2 : float
    Second stage :math:`A_2` response.
T_2 : float
    Second stage :math:`A_2` response.
D_2 : float
    Second stage :math:`D_2` response.
"""


def XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2, sigma=300):
    """
    Computes the *ATD (1995)* colour vision model correlates.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix of test sample / stimulus in domain
        [0, 100].
    XYZ_0 : array_like, (3,)
        *CIE XYZ* colourspace matrix of reference white in domain [0, 100].
    Y_0 : float
        Absolute adapting field luminance in :math:`cd/m^2`.
    k_1 : float
        Application specific weight :math:`k_1`.
    k_2 : float
        Application specific weight :math:`k_2`.
    sigma : float
        Constant :math:`\sigma` varied to predict different types of data.

    Returns
    -------
    ATD95_Specification
        *ATD (1995)* colour vision model specification.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 100].
    -   Input *CIE XYZ_0* colourspace matrix is in domain [0, 100].
    -   For unrelated colors, there is only self-adaptation, and :math:`k_1` is
        set to 1.0 while :math:`k_2` is set to 0.0. For related colors such as
        typical colorimetric applications, :math:`k_1` is set to 0.0 and
        :math:`k_2` is set to a value between 15 and 50 *Guth (1995)*.

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_0 = np.array([95.05, 100.00, 108.88])
    >>> Y_0 = 318.31
    >>> k_1 = 0.0
    >>> k_2 = 50.0
    >>> colour.XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2)
    ATD95_Specification(H=1.9089869677948668, Br=0.1814003693517946, C=1.2064060487501733, A_1=0.17879314415234579, T_1=0.028694273266804574, D_1=0.010758451876359426, A_2=0.019218221450414588, T_2=0.020537744425618377, D_2=0.010758451876359426)
    """

    XYZ = luminance_to_retinal_illuminance(XYZ, Y_0)
    XYZ_0 = luminance_to_retinal_illuminance(XYZ_0, Y_0)

    # Computing adaptation model.
    LMS = XYZ_to_LMS_ATD95(XYZ)
    XYZ_a = k_1 * XYZ + k_2 * XYZ_0
    LMS_a = XYZ_to_LMS_ATD95(XYZ_a)

    LMS_g = LMS * (sigma / (sigma + LMS_a))

    # Computing opponent colour dimensions.
    A_1, T_1, D_1, A_2, T_2, D_2 = get_opponent_colour_dimensions(LMS_g)

    # -------------------------------------------------------------------------
    # Computing the correlate of *brightness* :math:`Br`.
    # -------------------------------------------------------------------------
    brightness = (A_1 ** 2 + T_1 ** 2 + D_1 ** 2) ** 0.5

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`C`.
    # -------------------------------------------------------------------------
    saturation = (T_2 ** 2 + D_2 ** 2) ** 0.5 / A_2

    # -------------------------------------------------------------------------
    # Computing the *hue* :math:`H`.
    # -------------------------------------------------------------------------
    hue = T_2 / D_2

    return ATD95_Specification(hue,
                               brightness,
                               saturation,
                               A_1,
                               T_1,
                               D_1,
                               A_2,
                               T_2,
                               D_2)


def luminance_to_retinal_illuminance(XYZ, absolute_adapting_field_luminance):
    """
    Converts from luminance in :math:`cd/m^2` to retinal illuminance in
    trolands.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.

    absolute_adapting_field_luminance : float
        Absolute adapting field luminance in :math:`cd/m^2`.

    Returns
    -------
    ndarray
        Converted *CIE XYZ* colourspace matrix in trolands.

    Examples
    --------
    >>> XYZ = np.array([ 19.01,  20.  ,  21.78])
    >>> Y_0 = 318.31
    >>> colour.appearance.atd95.luminance_to_retinal_illuminance(XYZ, Y_0)
    array([ 479.44459244,  499.31743137,  534.56316731])
    """

    return 18. * (absolute_adapting_field_luminance * XYZ / 100.) ** 0.8


def XYZ_to_LMS_ATD95(XYZ):
    """
    Converts from *CIE XYZ* colourspace to *LMS* cone responses.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *LMS* cone responses.

    Examples
    --------
    >>> XYZ = np.array([ 19.01,  20.  ,  21.78])
    >>> Y_0 = 318.31
    >>> colour.appearance.atd95.luminance_to_retinal_illuminance(XYZ, Y_0)
    array([ 479.44459244,  499.31743137,  534.56316731])
    """

    X, Y, Z = np.ravel(XYZ)

    L = ((0.66 * (0.2435 * X + 0.8524 * Y - 0.0516 * Z)) ** 0.7) + 0.024
    M = ((-0.3954 * X + 1.1642 * Y + 0.0837 * Z) ** 0.7) + 0.036
    S = ((0.43 * (0.04 * Y + 0.6225 * Z)) ** 0.7) + 0.31

    return np.array([L, M, S])


def get_opponent_colour_dimensions(LMS_g):
    """
    Returns opponent colour dimensions from given post adaptation cone signals
    matrix.

    Parameters
    ----------
    LMS_g : array_like, (3,)
        Post adaptation cone signals matrix.

    Returns
    -------
    tuple
        Opponent colour dimensions.

    Examples
    --------
    >>> LMS_g = np.array([6.95457922, 7.08945043, 6.44069316])
    >>> colour.appearance.atd95.get_opponent_colour_dimensions(LMS_g)
    (0.17879314415234579, 0.028694273266804574, 0.010758451876359426, 0.019218221450414588, 0.020537744425618377, 0.010758451876359426)
    """

    L_g, M_g, S_g = LMS_g

    A_1i = 3.57 * L_g + 2.64 * M_g
    T_1i = 7.18 * L_g - 6.21 * M_g
    D_1i = -0.7 * L_g + 0.085 * M_g + S_g
    A_2i = 0.09 * A_1i
    T_2i = 0.43 * T_1i + 0.76 * D_1i
    D_2i = D_1i

    A_1 = get_final_response(A_1i)
    T_1 = get_final_response(T_1i)
    D_1 = get_final_response(D_1i)
    A_2 = get_final_response(A_2i)
    T_2 = get_final_response(T_2i)
    D_2 = get_final_response(D_2i)

    return A_1, T_1, D_1, A_2, T_2, D_2


def get_final_response(value):
    """
    Returns the final response of given opponent colour dimension.

    Parameters
    ----------
    value : float
         Opponent colour dimension.

    Returns
    -------
    float
        Final response of opponent colour dimension.

    Examples
    --------
    >>> colour.appearance.atd95.get_final_response(43.54399695501678)
    0.17879314415234579
    """

    return value / (200 + abs(value))

