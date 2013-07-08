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

model = MGMM(trainingSet)
trainingSet.setParent(model)

trainingSet.set_dimension_gesture(gdim)
trainingSet.set_dimension_sound(sdim)

model.set_nbMixtureComponents(6)
model.set_covarianceOffset(0.005)
model.set_EM_maxLogLikelihoodPercentChg(0.001)

model.initModelParameters()

for t in range(T_ref):
    trainingSet.recordPhrase(0, ref[t,:])

model.train()

model.dump()

result = np.zeros(T_test)
for t in range(T_test):
    prob, res = model.play(test[t,:], sdim)
    result[t] = res

plt.figure(0)
plt.plot(test, np.sin(test), test, result)

# trans_img = np.genfromtxt("test.txt")
# plt.figure(1)
# plt.imshow(trans_img, interpolation='nearest')

plt.show()