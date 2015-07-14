#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.volume.rgb` module.

Notes
-----
The MonteCarlo sampling based unit tests are assuming that
:func:`np.random.RandomState` definition will return the same sequence no
matter which *OS* or *Python* version is used. There is however no formal
promise about the *prng* sequence reproducibility of either *Python or *Numpy*
implementations:

Laurent. (2012). Reproducibility of python pseudo-random numbers across systems
and versions? Retrieved January 20, 2015, from
http://stackoverflow.com/questions/8786084/reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions  # noqa
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import (
    ACES_2065_1_COLOURSPACE,
    REC_2020_COLOURSPACE,
    REC_709_COLOURSPACE)
from colour.volume import (
    RGB_colourspace_limits,
    RGB_colourspace_volume_MonteCarlo,
    RGB_colourspace_volume_coverage_MonteCarlo,
    RGB_colourspace_pointer_gamut_coverage_MonteCarlo,
    RGB_colourspace_visible_spectrum_coverage_MonteCarlo,
    is_within_pointer_gamut)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_colourspaceLimits',
           'TestRGB_colourspaceVolumeMonteCarlo',
           'TestRGB_colourspace_volume_coverage_MonteCarlo',
           'TestRGB_colourspacePointerGamutCoverageMonteCarlo',
           'TestRGB_colourspaceVisibleSpectrumCoverageMonteCarlo']


class TestRGB_colourspaceLimits(unittest.TestCase):
    """
    Defines :func:`colour.volume.rgb.RGB_colourspace_limits` definition unit
    tests methods.
    """

    def test_RGB_colourspace_limits(self):
        """
        Tests :func:`colour.volume.rgb.RGB_colourspace_limits` definition.
        """

        np.testing.assert_almost_equal(
            RGB_colourspace_limits(REC_709_COLOURSPACE),
            np.array([[0., 100.],
                      [-79.22637417, 94.66574917],
                      [-114.78462716, 96.71351991]]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_colourspace_limits(REC_2020_COLOURSPACE),
            np.array([[0., 100.],
                      [-159.61691594, 127.33819164],
                      [-129.73792222, 142.12971261]]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_colourspace_limits(ACES_2065_1_COLOURSPACE),
            np.array([[-79.44256015, 103.30554311],
                      [-461.8341805, 176.445741],
                      [-309.6704667, 184.8212395]]),
            decimal=7)


class TestRGB_colourspaceVolumeMonteCarlo(unittest.TestCase):
    """
    Defines :func:`colour.volume.rgb.RGB_colourspace_volume_MonteCarlo`
    definition unit tests methods.
    """

    def test_RGB_colourspace_volume_MonteCarlo(self):
        """
        Tests :func:`colour.volume.rgb.RGB_colourspace_volume_MonteCarlo`
        definition.
        """

        self.assertEquals(
            RGB_colourspace_volume_MonteCarlo(
                REC_709_COLOURSPACE,
                10e3,
                random_state=np.random.RandomState(2),
                processes=1),
            859500.0)


class TestRGB_colourspace_volume_coverage_MonteCarlo(unittest.TestCase):
    """
    Defines
    :func:`colour.volume.rgb.RGB_colourspace_volume_coverage_MonteCarlo`
    definition unit tests methods.
    """

    def test_RGB_colourspace_volume_coverage_MonteCarlo(self):
        """
        Tests
        :func:`colour.volume.rgb.RGB_colourspace_volume_coverage_MonteCarlo`  # noqa
        definition.
        """

        np.testing.assert_almost_equal(
            RGB_colourspace_volume_coverage_MonteCarlo(
                REC_709_COLOURSPACE,
                is_within_pointer_gamut,
                10e3,
                random_state=np.random.RandomState(2)),
            83.02013422818791,
            decimal=7)


class TestRGB_colourspacePointerGamutCoverageMonteCarlo(unittest.TestCase):
    """
    Defines
    :func:`colour.volume.rgb.RGB_colourspace_pointer_gamut_coverage_MonteCarlo`
    definition unit tests methods.
    """

    def test_RGB_colourspace_pointer_gamut_coverage_MonteCarlo(self):
        """
        Tests
        :func:`colour.volume.rgb.RGB_colourspace_pointer_gamut_coverage_MonteCarlo`  # noqa
        definition.
        """

        np.testing.assert_almost_equal(
            RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
                REC_709_COLOURSPACE,
                10e3,
                random_state=np.random.RandomState(2)),
            83.02013422818791,
            decimal=7)


class TestRGB_colourspaceVisibleSpectrumCoverageMonteCarlo(unittest.TestCase):
    """
    Defines
    :func:`colour.volume.rgb.RGB_colourspace_visible_spectrum_coverage_MonteCarlo`  # noqa
    definition unit tests methods.
    """

    def test_RGB_colourspace_visible_spectrum_coverage_MonteCarlo(self):
        """
        Tests
        :func:`colour.volume.rgb.RGB_colourspace_visible_spectrum_coverage_MonteCarlo`  # noqa
        definition.
        """

        np.testing.assert_almost_equal(
            RGB_colourspace_visible_spectrum_coverage_MonteCarlo(
                REC_709_COLOURSPACE,
                10e3,
                random_state=np.random.RandomState(2)),
            36.48383937316356,
            decimal=7)


if __name__ == '__main__':
    unittest.main()
