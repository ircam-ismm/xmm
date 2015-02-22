#!/usr/bin/env python
# encoding: utf-8
"""
utility.py

Utilities for accessing and plotting model parameters in Python

Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
author: Jules Francoise <jules.francoise@ircam.fr>
"""


import numpy as np
import matplotlib.pyplot as plt
import xmm
import json
from matplotlib.patches import Ellipse
from scipy.stats import chi2


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


def compute_ellipse(mean, covariance, axis1, axis2):
    """ Compute the parameters of the ellipse representing a
    Gaussian distribution with dimension >= 2

    Args:
        mean -- mean of the Gaussian distribution
        covariance -- covariance of the Gaussian distribution
        axis1 -- 1st axis for plotting (x)
        axis2 -- 2nd axis for plotting (y)

    Returns:
        ellipse -- Ellipse representing the Gaussian distribution

    Raises:
        Exception -- if error with dimensions
    """
    dimension = mean.shape[0]
    if dimension < 2:
        raise Exception("Gaussian must have dimension >= 2")
    if axis1 >= dimension or axis2 >= dimension:
        raise Exception("index out of bounds")
    mean2d = mean[[axis1, axis2]]
    cov2d = covariance[[axis1, axis1, axis2, axis2],
                       [axis1, axis2, axis1, axis2]]\
                       .reshape((2, 2))
    eigenVal, eigenVec = np.linalg.eig(cov2d)
    theta = np.arctan(np.real(eigenVec[1, 0]) / np.real(eigenVec[0, 0]))
    # Ellipse for 95% confidence interval
    print mean2d[0], mean2d[1], ellipse_xmm.widthsqrt(chi2.ppf(0.95, 2)*eigenVal[1]), theta
    ellipse = Ellipse(xy=mean2d,
                  width=2*np.sqrt(chi2.ppf(0.95, 2)*eigenVal[0]),
                  height=2*np.sqrt(chi2.ppf(0.95, 2)*eigenVal[1]),
                  angle=theta*180/np.pi)
    ellipse.set_alpha(0.5)
    return ellipse


def gmm_plot_gaussians(gmm, axis1, axis2, axes=None, color='b'):
    """ Plot the Gaussian Parameters of a GMM

    Args:
        gmm_dict -- python dict describing a gmm
        axis1 -- 1st axis for plotting
        axis2 -- 2nd axis for plotting

    Raises:
        Exception -- if error with dimensions
    """
    dimension = gmm.dimension()
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

