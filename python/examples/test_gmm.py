#!/usr/bin/env python
# encoding: utf-8
"""
test_gmm.py

Simple Test File for GMMs

Contact:
- Jules Francoise <jules.francoise@ircam.fr>

This code has been initially authored by Jules Francoise
<http://julesfrancoise.com> during his PhD thesis, supervised by Frederic
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


def create_training_set():
    """ Create the training set for building a simple GMM recognizer

    The training data originates from the help patch: "mubu.gmm.maxhelp".

    Returns:
        training_set -- Unimodal Training Set
    """
    # Create the training set
    training_set = xmm.TrainingSet()
    training_set.dimension.set(6) # dimension of data in this example
    # Record data phrases
    for i in range(3):
        phrase = np.genfromtxt('data/gmm_training_data{}.txt'.format(i+1))
        training_set.addPhrase(i, str(i+1))
        for frame in phrase:
            # Append data frame to the phrase i
            training_set.getPhrase(i).record(frame)
    return training_set


def gmm_train(training_set, num_gaussians=1, varianceoffset=[1., 0.01]):
    """ Create and Traing a GMM from the given training set

    Args:
        training_set -- Unimodal Training Set
        num_gaussians -- Number of Gaussian Components
        varianceoffset -- rel/abs variance offset

    Returns:
        gmm -- Trained GMM group
    """
    # Create a GMM Group (handles multiples labels for recognition)
    gmm = xmm.GMM()
    # Set parameters
    gmm.configuration.gaussians.set(num_gaussians)
    gmm.configuration.relative_regularization.set(varianceoffset[0])
    gmm.configuration.absolute_regularization.set(varianceoffset[1])
    # Train all models
    gmm.train(training_set)
    return gmm


def gmm_test_recognition(gmm, likelihood_window=20):
    """ Simple Test Function for Gaussian Mixture Models.
    The data originates from the help patch: "mubu.gmm.maxhelp".

    Args:
        gmm -- trained GMM Group
        likelihood_window -- size of the smoothing window for recognition
    """
    # read test data (concatenation of 3 test examples labeled 1, 2, 3)
    test_data = np.genfromtxt('data/gmm_test_data1.txt')
    test_data = np.vstack((test_data,
                          np.genfromtxt('data/gmm_test_data2.txt')))
    test_data = np.vstack((test_data,
                          np.genfromtxt('data/gmm_test_data3.txt')))
    # Set Size of the likelihood Window (samples)
    gmm.shared_parameters.likelihood_window.set(likelihood_window)
    # Initialize performance phase
    gmm.reset()
    # Create likelihood arrays for recognition
    instantaneous_likelihoods = np.zeros((test_data.shape[0], gmm.size()))
    normalized_likelihoods = np.zeros((test_data.shape[0], gmm.size()))
    log_likelihoods = np.zeros((test_data.shape[0], gmm.size()))
    # Performance: Play test data and record the likelihoods of the modes
    for i in range(test_data.shape[0]):
        gmm.filter(test_data[i, :])
        instantaneous_likelihoods[i, :] = np.array(gmm.results.instant_likelihoods)
        normalized_likelihoods[i, :] = np.array(gmm.results.smoothed_normalized_likelihoods)
        log_likelihoods[i, :] = np.array(gmm.results.smoothed_log_likelihoods)
    # Plot the likelihoods over time for the test phase
    plt.figure()
    plt.subplot(311)
    plt.plot(instantaneous_likelihoods)
    plt.title("Instantaneous Likelihood of Each Model Over time")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Likelihood")
    plt.legend(("model 1", "model 2", "model 3"))
    plt.subplot(312)
    plt.plot(normalized_likelihoods)
    plt.title("Normalized Smoothed Likelihood of Each Model Over time")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Normalized Likelihood")
    plt.legend(("model 1", "model 2", "model 3"))
    plt.subplot(313)
    plt.plot(log_likelihoods)
    plt.title("Smoothed Log-Likelihood of Each Model Over time")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Log-Likelihood")
    plt.legend(("model 1", "model 2", "model 3"))
    # plt.show()
    return log_likelihoods


if __name__ == '__main__':
    TRAINING_SET = create_training_set()
    GMM_MODEL = gmm_train(TRAINING_SET)
    LIKELIHOOD_WINDOW = 1 # 50
    LOG_LIKELIHOODS1 = gmm_test_recognition(GMM_MODEL, LIKELIHOOD_WINDOW)
    LIKELIHOOD_WINDOW = 40 # 50
    LOG_LIKELIHOODS2 = gmm_test_recognition(GMM_MODEL, LIKELIHOOD_WINDOW)
    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(LOG_LIKELIHOODS1)
    plt.title("Smoothed Log-Likelihood of Each Model Over time ($L_W = 1$)")
    # plt.xlabel("Time (Samples)")
    plt.ylabel("Log-likelihood")
    plt.legend(("model 1", "model 2", "model 3"), loc='best')
    ax2 = plt.subplot(212)
    plt.plot(LOG_LIKELIHOODS2)
    plt.title("Smoothed Log-Likelihood of Each Model Over time ($L_W = 30$)")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Log-likelihood")
    plt.legend(("model 1", "model 2", "model 3"), loc='best')
    plt.show()



