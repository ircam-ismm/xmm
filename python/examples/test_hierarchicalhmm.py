#!/usr/bin/env python
# encoding: utf-8
"""
test_hierarchicalhmm.py

Test File for Recognition with the Hierarchical HMM

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


def test_hhmm():
    """ Simple Test Function for Recognition with Hierarchical Hidden Markov Models.
    The data originates from the example patch: "hhmm_leapmotion_recognition.maxpat".
    The data represents the XYZ Speed of the hand extracted from the leapmotion, rescaled
    (divided by 1000) and smoothed (moving average filter).
    """
    # Create a single HMM (group of GMMs running in parallel for recognition)
    hhmm = xmm.HierarchicalHMM()
    # Read trained model from Json file
    hhmm.readFile('data/hhmm_model.json')
    # read test data (concatenation of 1 example of each of the 3 classes)
    test_data = np.genfromtxt('data/hhmm_test_data1.txt')
    test_data = np.vstack((test_data, np.genfromtxt('data/hhmm_test_data2.txt')))
    test_data = np.vstack((test_data, np.genfromtxt('data/hhmm_test_data3.txt')))
    # Initialize performance phase
    hhmm.performance_init()
    # Create arrays for likelihoods
    instantaneous_likelihoods = np.zeros((test_data.shape[0], hhmm.size()))
    normalized_likelihoods = np.zeros((test_data.shape[0], hhmm.size()))
    log_likelihoods = np.zeros((test_data.shape[0], hhmm.size()))
    progress = np.zeros((test_data.shape[0]))
    # Performance: Play test data and record the likelihoods of the modes
    for i in range(test_data.shape[0]):
        hhmm.performance_update(xmm.vectorf(test_data[i, :]))
        log_likelihoods[i, :] = np.array(hhmm.results_log_likelihoods)
        instantaneous_likelihoods[i, :] = np.array(hhmm.results_instant_likelihoods)
        normalized_likelihoods[i, :] = np.array(hhmm.results_normalized_likelihoods)
        progress[i] = hhmm.models[hhmm.results_likeliest].results_progress
        # Note: you could extract alphas and time progression as for HMM, for each model. E.g. : 
        # print np.array(hhmm.models[hhmm.results_likeliest].alpha)
    # Plot the likelihoods over time for the test phase
    plt.figure()
    plt.subplot(411)
    plt.plot(instantaneous_likelihoods)
    plt.title("Instantaneous Likelihood of Each Model Over time")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Likelihood")
    plt.legend(("model 1", "model 2", "model 3"))
    plt.subplot(412)
    plt.plot(normalized_likelihoods)
    plt.title("Normalized Smoothed Likelihood of Each Model Over time")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Normalized Likelihood")
    plt.legend(("model 1", "model 2", "model 3"))
    plt.subplot(413)
    plt.plot(log_likelihoods)
    plt.title("Smoothed Log-Likelihood of Each Model Over time")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Log-Likelihood")
    plt.legend(("model 1", "model 2", "model 3"))
    plt.subplot(414)
    plt.plot(progress)
    plt.title("Normalized progression with the likeliest model")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Normalized Progress")
    plt.show()


if __name__ == '__main__':
    test_hhmm()