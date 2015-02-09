#!/usr/bin/env python
# encoding: utf-8
"""
test_gmm.py

Simple Test File for GMMs

Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
author: Jules Francoise <jules.francoise@ircam.fr>
"""

import numpy as np
import matplotlib.pyplot as plt
import mhmm
import thesis


def create_training_set():
    """ Create the training set for building a simple GMM recognizer
    
    The training data originates from the help patch: "mubu.gmm.maxhelp".

    Returns:
        training_set -- Unimodal Training Set
    """
    # Create the training set
    training_set = mhmm.TrainingSet()
    training_set.set_dimension(6) # dimension of data in this example
    # Record data phrases
    for i in range(3):
        phrase = np.genfromtxt('gmm_test_data{}.txt'.format(i+1))
        for frame in phrase:
            # Append data frame to the phrase i
            training_set.recordPhrase(i, frame)
        # Set phrase label
        training_set.setPhraseLabel(i, mhmm.Label(i+1))
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
    gmm = mhmm.GMMGroup()
    # Set pointer to the training set
    gmm.set_trainingSet(training_set)
    # Set parameters
    gmm.set_nbMixtureComponents(num_gaussians)
    gmm.set_varianceOffset(varianceoffset[0], varianceoffset[1])
    # Train all models
    gmm.train()
    print "model 1: trained in ", gmm.models[mhmm.Label(1)].trainingNbIterations, \
            "iterations, loglikelihood = ", gmm.models[mhmm.Label(1)].trainingLogLikelihood
    print "model 2: trained in ", gmm.models[mhmm.Label(2)].trainingNbIterations, \
            "iterations, loglikelihood = ", gmm.models[mhmm.Label(2)].trainingLogLikelihood
    print "model 3: trained in ", gmm.models[mhmm.Label(3)].trainingNbIterations, \
            "iterations, loglikelihood = ", gmm.models[mhmm.Label(3)].trainingLogLikelihood
    return gmm


def gmm_test_recognition(gmm, likelihood_window=20):
    """ Simple Test Function for Gaussian Mixture Models.
    The data originates from the help patch: "mubu.gmm.maxhelp".

    Args:
        gmm -- trained GMM Group
        likelihood_window -- size of the smoothing window for recognition
    """
    # read test data (concatenation of 3 test examples labeled 1, 2, 3)
    test_data = np.genfromtxt('gmm_test_data1.txt')
    test_data = np.vstack((test_data, np.genfromtxt('gmm_test_data2.txt')))
    test_data = np.vstack((test_data, np.genfromtxt('gmm_test_data3.txt')))
    # Set Size of the likelihood Window (samples)
    gmm.set_likelihoodwindow(likelihood_window)
    # Initialize performance phase
    gmm.performance_init()
    # Create likelihood arrays for recognition
    instantaneous_likelihoods = np.zeros((test_data.shape[0], gmm.size()))
    normalized_likelihoods = np.zeros((test_data.shape[0], gmm.size()))
    log_likelihoods = np.zeros((test_data.shape[0], gmm.size()))
    # Performance: Play test data and record the likelihoods of the modes
    for i in range(test_data.shape[0]):
        gmm.performance_update(mhmm.vectorf(test_data[i, :]))
        instantaneous_likelihoods[i, :] = np.array(gmm.results_instant_likelihoods)
        normalized_likelihoods[i, :] = np.array(gmm.results_normalized_likelihoods)
        log_likelihoods[i, :] = np.array(gmm.results_log_likelihoods)
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
    # GMM_MODEL = mhmm.GMMGroup()
    # GMM_MODEL.readFile('gmm_model.json')
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
    thesis.cleanAxes(ax1)
    thesis.cleanAxes(ax2)
    plt.show()



