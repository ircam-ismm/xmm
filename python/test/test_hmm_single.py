#!/usr/bin/env python
# encoding: utf-8
"""
test_hmm_single.py

Test file for a single HMM

Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
author: Jules Francoise <jules.francoise@ircam.fr>
"""

import numpy as np
import matplotlib.pyplot as plt
import mhmm


def test_hmm():
    """ Simple Test Function for Hidden Markov Models.
    The data originates from the example patch: "hhmm_following.maxpat".
    """
    # Create a single HMM (group of GMMs running in parallel for recognition)
    hmm = mhmm.HMM(mhmm.NONE, None, 10, 1)
    # Read trained model from Json file
    hmm.readFile('hmm_single_model_1.json')
    # read test data (concatenation of the 3 training examples)
    test_data = np.genfromtxt('hmm_single_data.txt')
    # Initialize performance phase
    hmm.performance_init()
    # Create likelihood array for recognition
    likelihoods = np.zeros((test_data.shape[0], 1))
    log_likelihoods = np.zeros((test_data.shape[0], 1))
    timeprogression = np.zeros((test_data.shape[0], 1))
    alphas = np.zeros((test_data.shape[0], hmm.get_nbStates()))
    # Performance: Play test data and record the likelihoods of the modes
    for i in range(test_data.shape[0]):
        hmm.performance_update(mhmm.vectorf(test_data[i, :]))
        log_likelihoods[i, :] = np.array(hmm.results_log_likelihood)
        likelihoods[i, :] = np.array(hmm.results_instant_likelihood)
        timeprogression[i, :] = np.array(hmm.results_progress)
        alphas[i, :] = np.array(hmm.alpha)
    np.savetxt("hmm_results_likelihoods.txt", likelihoods)
    np.savetxt("hmm_results_loglikelihoods.txt", log_likelihoods)
    np.savetxt("hmm_results_alphas.txt", alphas)
    np.savetxt("hmm_results_timeprogression.txt", timeprogression)
    # Plot the likelihoods over time for the test phase
    plt.figure()
    plt.subplot(311)
    plt.plot(log_likelihoods)
    plt.title("Log-Likelihood")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Log-Likelihood")
    plt.subplot(312)
    plt.plot(alphas)
    plt.title("State Probabilities")
    plt.xlabel("Time (Samples)")
    plt.ylabel("State Probabilities")
    plt.subplot(313)
    plt.plot(timeprogression)
    plt.title("Normalized Time Progression")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Normalized Time Progression")
    plt.show()


if __name__ == '__main__':
    test_hmm()