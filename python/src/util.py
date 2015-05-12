#!/usr/bin/env python
# encoding: utf-8
"""
utility.py

Utilities for accessing and plotting model parameters in Python

Contact:
- Jules Françoise <jules.francoise@ircam.fr>

This code has been initially authored by Jules Françoise
<http://julesfrancoise.com> during his PhD thesis, supervised by Frédéric
Bevilacqua <href="http://frederic-bevilacqua.net>, in the Sound Music
Movement Interaction team <http://ismm.ircam.fr> of the
STMS Lab - IRCAM, CNRS, UPMC (2011-2015).

Copyright (C) 2015 UPMC, Ircam-Centre Pompidou.

This File is part of XMM.

XMM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

XMM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with XMM.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
import matplotlib.pyplot as plt
import xmm
import json
from matplotlib.patches import Ellipse


def get_labels(model_group):
    """ Get the list of labels (int, string) from a model group

    Args:
        model_group -- model group (e.g. GMMGroup, HierarchicaHMM)
    """
    labels = []
    for lab in model_group.models.keys():
        labels.append(lab.getInt() if lab.type == 0 else lab.getSym())
    return labels


def get_model(model_group, label):
    """ Get a model from a group

    Args:
        model_group -- model group (e.g. GMMGroup, HierarchicaHMM)
        label -- label as int/string
    """
    return model_group.models[xmm.Label(label)]


def model2dict(model):
    """ Convert a model to a python dictionnary (parameters only)
    Uses the JSON I/O function

    Args:
        model -- xmm model or group
    """
    return dict(json.loads(model.__str__()))


def gmm_plot_gaussians(gmm, axis1, axis2, axes=None, color='b'):
    """ Plot the Gaussian Parameters of a GMM

    Args:
        gmm_dict -- python dict describing a gmm
        axis1 -- 1st axis for plotting
        axis2 -- 2nd axis for plotting

    Raises:
        Exception -- if error with dimensions
    """
    num_gaussians = gmm.get_nbMixtureComponents()
    ellipses = []
    for gaussian in range(num_gaussians):
        ellipse_xmm = gmm.components[gaussian].ellipse(axis1, axis2)
        ellipses.append(Ellipse(xy=[ellipse_xmm.x, ellipse_xmm.y],
                                width=ellipse_xmm.width,
                                height=ellipse_xmm.height,
                                angle=ellipse_xmm.angle*180/np.pi))
    if not axes:
        fig = plt.figure()
        axes = fig.add_subplot(111, aspect='equal')
    for ell in ellipses:
        ell.set_facecolor(color)
        ell.set_edgecolor((1, 1, 1, 0.))
        # ell.set_alpha(0.5)
        axes.add_artist(ell)
        ell.set_alpha(0.5)
    return axes


def test_gmm():
    """ Simple test Function For GMM Plotting
    """
    gmmgroup = xmm.GMMGroup()
    gmmgroup.readFile('test/gmm_model.json')
    labels = get_labels(gmmgroup)
    axes = None
    for i, label in enumerate(labels):
        gmm_params = model2dict(get_model(gmmgroup, label))
        axes = gmm_plot_gaussians(gmm_params, 2, 1, axes,
                                  plt.rcParams['axes.color_cycle'][i])
    axes.set_xlim(-2, 2)
    axes.set_ylim(-2, 2)
    plt.show()


if __name__ == '__main__':
    test_gmm()

