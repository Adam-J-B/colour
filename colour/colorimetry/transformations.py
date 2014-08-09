#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Matching Functions Transformations
=========================================

Defines various educational objects for colour matching functions
transformations:

-   :func:`RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs`
-   :func:`RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs`
-   :func:`RGB_10_degree_cmfs_to_LMS_10_degree_cmfs`
-   :func:`LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs`
"""

from __future__ import unicode_literals

import numpy as np

from colour.colorimetry import LMS_CMFS, RGB_CMFS, PHOTOPIC_LEFS

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs",
           "RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs",
           "RGB_10_degree_cmfs_to_LMS_10_degree_cmfs",
           "LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs"]


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts *Wright & Guild 1931 2 Degree RGB CMFs* colour matching functions
    into the *CIE 1931 2 Degree Standard Observer* colour matching functions.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *CIE 1931 2 Degree Standard Observer* spectral tristimulus values.

    See Also
    --------
    :attr:`colour.colorimetry.dataset.cmfs.RGB_CMFS`

    Notes
    -----
    -   Data for the *CIE 1931 2 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [1]  **Wyszecki & Stiles**,
            *Color Science - Concepts and Methods Data and Formulae -
            Second Edition*,
            Wiley Classics Library Edition, published 2000,
            ISBN-10: 0-471-39918-3,
            Pages 138, 139.

    Examples
    --------
    >>> colour.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)
    array([ 0.01135774,  0.004102  ,  0.        ])
    """

    cmfs = RGB_CMFS.get("Wright & Guild 1931 2 Degree RGB CMFs")
    r_bar, g_bar, b_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(
        wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, g_bar, b_bar):
        raise KeyError("'{0} nm' wavelength not available in '{1}' colour \
        matching functions with '{2}' shape!".format(
            wavelength, cmfs.name, cmfs.shape))

    r = r_bar / (r_bar + g_bar + b_bar)
    g = g_bar / (r_bar + g_bar + b_bar)
    b = b_bar / (r_bar + g_bar + b_bar)

    x = ((0.49000 * r + 0.31000 * g + 0.20000 * b) /
         (0.66697 * r + 1.13240 * g + 1.20063 * b))
    y = ((0.17697 * r + 0.81240 * g + 0.01063 * b) /
         (0.66697 * r + 1.13240 * g + 1.20063 * b))
    z = ((0.00000 * r + 0.01000 * g + 0.99000 * b) /
         (0.66697 * r + 1.13240 * g + 1.20063 * b))

    V = PHOTOPIC_LEFS.get("CIE 1924 Photopic Standard Observer").clone()
    V.align(*cmfs.shape)
    L = V.get(wavelength)

    x_bar = x / y * L
    y_bar = L
    z_bar = z / y * L

    return np.array([x_bar, y_bar, z_bar])


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *CIE 1964 10 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *CIE 1964 10 Degree Standard Observer* spectral tristimulus values.

    See Also
    --------
    :attr:`colour.colorimetry.dataset.cmfs.RGB_CMFS`

    Notes
    -----
    -   Data for the *CIE 1964 10 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [2]  **Wyszecki & Stiles**,
            *Color Science - Concepts and Methods Data and Formulae -
            Second Edition*,
            Wiley Classics Library Edition, published 2000,
            ISBN-10: 0-471-39918-3,
            Page 141.

    Examples
    --------
    >>> colour.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)
    array([  9.64321500e-03,   3.75263179e-03,  -4.10788300e-06])
    """

    cmfs = RGB_CMFS.get("Stiles & Burch 1959 10 Degree RGB CMFs")
    r_bar, g_bar, b_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(
        wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, g_bar, b_bar):
        raise KeyError("'{0} nm' wavelength not available in '{1}' colour \
        matching functions with '{2}' shape!".format(
            wavelength, cmfs.name, cmfs.shape))

    x_bar = 0.341080 * r_bar + 0.189145 * g_bar + 0.387529 * b_bar
    y_bar = 0.139058 * r_bar + 0.837460 * g_bar + 0.073316 * b_bar
    z_bar = 0.000000 * r_bar + 0.039553 * g_bar + 2.026200 * b_bar

    return np.array([x_bar, y_bar, z_bar])


def RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wavelength):
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *Stockman & Sharpe 10 Degree Cone Fundamentals*
    spectral sensitivity functions.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *Stockman & Sharpe 10 Degree Cone Fundamentals* spectral tristimulus
        values.

    Notes
    -----
    -   Data for the *Stockman & Sharpe 10 Degree Cone Fundamentals* already
        exists, this definition is intended for educational purpose.

    References
    ----------
    .. [3]  `CIE 170-1:2006 Fundamental Chromaticity Diagram with Physiological
            Axes - Part 1 <http://div1.cie.co.at/?i_ca_id=551&pubid=48>`_

    Examples
    --------
    >>> colour.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700)
    array([ 0.00528607,  0.00032528,  0.        ])
    """

    cmfs = RGB_CMFS.get("Stiles & Burch 1959 10 Degree RGB CMFs")
    r_bar, g_bar, z_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(
        wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, g_bar, z_bar):
        raise KeyError("'{0} nm' wavelength not available in '{1}' colour \
        matching functions with '{2}' shape!".format(
            wavelength, cmfs.name, cmfs.shape))

    l_bar = 0.192325269 * r_bar + 0.749548882 * g_bar + 0.0675726702 * z_bar
    g_bar = 0.0192290085 * r_bar + 0.940908496 * g_bar + 0.113830196 * z_bar
    z_bar = (0.0105107859 * g_bar + 0.991427669 * z_bar
             if wavelength <= 505 else 0.)

    return np.array([l_bar, g_bar, z_bar])


def LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts *Stockman & Sharpe 2 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 2 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *CIE 2012 2 Degree Standard Observer* spectral tristimulus values.

    Notes
    -----
    -   Data for the *CIE 2012 2 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [4]  http://www.cvrl.org/database/text/cienewxyz/cie2012xyz2.htm
            (Last accessed 25 June 2014)

    Examples
    --------
    >>> colour.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)
    array([ 0.01096778,  0.00419594,  0.        ])
    """

    cmfs = LMS_CMFS.get("Stockman & Sharpe 2 Degree Cone Fundamentals")
    l_bar, m_bar, s_bar = cmfs.l_bar.get(wavelength), cmfs.m_bar.get(
        wavelength), cmfs.s_bar.get(wavelength)
    if None in (l_bar, m_bar, s_bar):
        raise KeyError("'{0} nm' wavelength not available in '{1}' colour \
        matching functions with '{2}' shape!".format(
            wavelength, cmfs.name, cmfs.shape))

    x_bar = 1.94735469 * l_bar - 1.41445123 * m_bar + 0.36476327 * s_bar
    y_bar = 0.68990272 * l_bar + 0.34832189 * m_bar
    z_bar = 1.93485343 * s_bar

    return np.array([x_bar, y_bar, z_bar])


def LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts *Stockman & Sharpe 10 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 10 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *CIE 2012 10 Degree Standard Observer* spectral tristimulus values.

    Notes
    -----
    -   Data for the *CIE 2012 10 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [5]  http://www.cvrl.org/database/text/cienewxyz/cie2012xyz10.htm
            (Last accessed 25 June 2014)

    Examples
    --------
    >>> colour.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)
    array([ 0.00981623,  0.00377614,  0.        ])
    """

    cmfs = LMS_CMFS.get("Stockman & Sharpe 10 Degree Cone Fundamentals")
    l_bar, m_bar, s_bar = cmfs.l_bar.get(wavelength), cmfs.m_bar.get(
        wavelength), cmfs.s_bar.get(wavelength)
    if None in (l_bar, m_bar, s_bar):
        raise KeyError("'{0} nm' wavelength not available in '{1}' colour \
        matching functions with '{2}' shape!".format(
            wavelength, cmfs.name, cmfs.shape))

    x_bar = 1.93986443 * l_bar - 1.34664359 * m_bar + 0.43044935 * s_bar
    y_bar = 0.69283932 * l_bar + 0.34967567 * m_bar
    z_bar = 2.14687945 * s_bar

    return np.array([x_bar, y_bar, z_bar])