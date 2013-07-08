#XCode stuff
import os
os.chdir('/Users/francoise/Documents/Code/XCode/mhmm_new/python')

import numpy as np
from mhmm import *
import matplotlib.pyplot as plt

gdim = 1
sdim = 1
T_ref = 100
T_test = T_ref

ref = np.transpose(np.vstack((np.linspace(0,np.pi,T_ref), np.sin(np.linspace(0,np.pi,T_ref)))))
# test = np.transpose(np.vstack((np.linspace(0,np.pi/2,T_test), np.sin(np.linspace(0,np.pi/2,T_test)))))
test = np.linspace(0, np.pi,T_ref)
test.shape = (T_ref, 1)

trainingSet = mTrainingSet()

model = PolyMGMM(trainingSet)
trainingSet.setParent(model)

trainingSet.set_dimension_gesture(gdim)
trainingSet.set_dimension_sound(sdim)

model.set_nbMixtureComponents(3)
model.set_covarianceOffset(0.005)
model.set_EM_maxLogLikelihoodPercentChg(0.001)

model.initModelParameters()

for t in range(T_ref):
    trainingSet.recordPhrase(0, ref[t,:])
    trainingSet.setClassLabel(0, 1)

for t in range(T_ref):
    trainingSet.recordPhrase(1, ref[t,:]-0.1)
    trainingSet.setClassLabel(1, 2)

nbIt = model.train()
print "model ", 1, "trained after ", nbIt[1], "iterations"
print "model ", 2, "trained after ", nbIt[2], "iterations\n"

result = np.zeros(T_test)
probs = np.zeros((T_test, 2))
for t in range(T_test):
    res, prob = model.play(test[t,:], sdim, 2)
    probs[t,:] = prob
    result[t] = res

print probs

plt.figure()
plt.plot(test, np.sin(test), test, result, test, np.sin(test)-0.1)

# trans_img = np.genfromtxt("test.txt")
# plt.figure(1)
# plt.imshow(trans_img, interpolation='nearest')

plt.show()