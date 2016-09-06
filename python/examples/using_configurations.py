#!/usr/bin/env python
# encoding: utf-8
"""
Example of using configuration to specify class parameters

Contact:
- Jules Francoise <jfrancoi@sfu.ca>

Copyright (C) 2015 UPMC, Ircam-Centre Pompidou.

This File is part of XMM.
"""

# import matplotlib.pyplot as plt
import numpy as np
import xmm


# Create the training set
training_set = xmm.TrainingSet()
training_set.dimension.set(3)
labels = ['a', 'b', 'c']

# Create the model and set the default configuration
hhmm = xmm.HierarchicalHMM()
hhmm.configuration.states.set(10)  # set default number of state per class

# Record each phrase (random) to the training set and give each class a number
# of states proportional to the phrase length
for i in range(3):
    data = np.genfromtxt('data/hhmm_test_data%i.txt' % (i + 1))
    training_set.addPhrase(i, labels[i])
    [training_set.getPhrase(i).record(frame) for frame in data]
    num_states = len(data) // 10
    # Set the number of states for each class
    hhmm.configuration[labels[i]].states.set(num_states)

# Train the model
hhmm.train(training_set)

# Print model parameters after training
for label in labels:
    print('The HMM for for class %s has %i states' %
          (labels[i], hhmm.models[label].parameters.states.get()))
