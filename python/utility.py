from numpy import *
from matplotlib.patches import Ellipse
from scipy.linalg import eig
from matplotlib.pyplot import *

colorIndex = 0

class MGMM_params():
    """MGMM_params: Dummy class to store the parameters of a MGMM estimated by EM."""
    
    def __init__(self):
        self.dimension_gesture = 0
        self.dimension_sound = 0
        self.nbMixtures = 0
        self.covarianceOffset = 0
        self.EM_stopCriterion = 0
        self.EM_maxSteps = 0
        self.EM_percentchg = 0
        self.likelihoodWindow = 0
        self.mixtureCoeffs = None
        self.mean = None
        self.covariance = None
    
    def dump(self):
        print "=============================================="
        print "== MULTIMODAL GMM =="
        print "=============================================="
        print "Dimensions : ", self.dimension_gesture, self.dimension_sound
        print "Number of mixture components = ", self.nbMixtures
        print "covariance offset = ", self.covarianceOffset
        print "Em criterion: ", self.EM_stopCriterion, self.EM_maxSteps,self.EM_percentchg
        print "likelihoodWindow = ", self.likelihoodWindow
        print "mixture Coefficients : "
        print self.mixtureCoeffs
        print "Mean : "
        print self.mean
        print "Covariance :"
        print self.covariance


class MHMM_params():
    """MHMM_params: Dummy class to store the parameters of a MHMM estimated by EM."""
    
    def __init__(self):
        self.nbStates = 0
        self.dimension_gesture = 0
        self.dimension_sound = 0
        self.nbMixtures = 0
        self.covarianceOffset = 0
        self.EM_stopCriterion = 0
        self.EM_maxSteps = 0
        self.EM_percentchg = 0
        self.likelihoodWindow = 0
        self.prior = None
        self.transition = None
        self.states = []
    
    def dump(self):
        print "==============================================\n== MULTIMODAL HMM ==\n==============================================\n"
        print "Number of states: ", self.nbStates
        print "Dimensions : ", self.dimension_gesture, self.dimension_sound
        print "Number of mixture components = ", self.nbMixtures
        print "covariance offset = ", self.covarianceOffset
        print "Em criterion: ", self.EM_stopCriterion, self.EM_maxSteps, self.EM_percentchg
        print "prior:"
        print self.prior
        print "transition:"
        print self.transition
        for i in range(self.nbStates):
            print "------------------------------------------\n== State", i, " :\n------------------------------------------"
            print "mixture Coefficients : "
            print self.states[i].mixtureCoeffs
            print "Mean : "
            print self.states[i].mean
            print "Covariance :"
            print self.states[i].covariance

def read_MultimodalGMM(file):
    # Get EM Stop criterion
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    s = s.strip().split()
    EM_stopCriterion  = int(s[0])
    EM_maxSteps  = int(s[1])
    EM_percentchg  = float(s[2])
    
    # get likelihoodBufferSize
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    likelihoodWindow = int(s)
    
    # Get Dimensions
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    s = s.strip().split()
    dimension_gesture  = int(s[0])
    dimension_sound  = int(s[1])
    dimension_tot = dimension_gesture + dimension_sound
    
    # Get number of mixtures
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    nbMixtures = int(s)
    
    # Get covariance offset
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    covarianceOffset = float(s)
    
    # Get mixture coefficients
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    
    mixtureCoeffs = zeros(nbMixtures)
    s = s.strip().split()
    for c in range(nbMixtures):
        mixtureCoeffs[c] = float(s[c])
    
    # Get means
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    
    mean = zeros((nbMixtures, dimension_tot))
    for c in range(nbMixtures):
        s = s.strip().split()
        for d in range(dimension_tot):
            mean[c, d] = float(s[d])
        s = file.readline()
    
    # Get covariances
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    
    covariance = zeros((nbMixtures, dimension_tot, dimension_tot))
    for c in range(nbMixtures):
        for d1 in range(dimension_tot):
            s = s.strip().split()
            for d2 in range(dimension_tot):
                covariance[c, d1, d2] = float(s[d2])
            s = file.readline()
    
    model = MGMM_params()
    model.dimension_gesture = dimension_gesture
    model.dimension_sound = dimension_sound
    model.nbMixtures = nbMixtures
    model.covarianceOffset = covarianceOffset
    model.EM_stopCriterion = EM_stopCriterion
    model.EM_maxSteps = EM_maxSteps
    model.EM_percentchg = EM_percentchg
    model.likelihoodWindow = likelihoodWindow
    model.mixtureCoeffs = mixtureCoeffs
    model.mean = mean
    model.covariance = covariance
    
    return model


def read_PolyMultimodalGMM(file):
    # Read Number of models
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    nbModels  = int(s)
    
    # Read Play mode
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    playMode  = int(s)
    
    # Read  Reference Model ==> Nowhere
    # ------------------------------
    read_MultimodalGMM(file)
    
    # Read Models
    # ------------------------------
    polyMGMM = []
    for i in range(nbModels):
        # Read model index
        # ------------------------------
        s =  file.readline()
        while s[0] == '#':
            s =  file.readline()
        # print "model index:", int(s)
        polyMGMM.append(read_MultimodalGMM(file))
    
    return polyMGMM


def read_MultimodalHMM(file):
    # Get EM Stop criterion
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    s = s.strip().split()
    EM_stopCriterion  = int(s[0])
    EM_maxSteps  = int(s[1])
    EM_percentchg  = float(s[2])
    
    # get likelihoodBufferSize
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    likelihoodWindow = int(s)
    
    # Get Dimensions
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    s = s.strip().split()
    dimension_gesture  = int(s[0])
    dimension_sound  = int(s[1])
    dimension_tot = dimension_gesture + dimension_sound
    
    # Get Number of states
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    nbStates = int(s)
    
    prior = zeros(nbStates)
    transition = zeros((nbStates, nbStates))
    
    # Get Transition Mode (Ergodic / Left-Right)
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    transitionMode = int(s)
    
    # Get number of mixtures
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    nbMixtures = int(s)
    
    # Get covariance offset
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    covarianceOffset = float(s)
    
    # Get Prior
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    s = s.strip().split()
    for i in range(nbStates):
        prior[i] = float(s[i])
    
    # Get Transition Matrix
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    for i in range(nbStates):
        s = s.strip().split()
        for j in range (nbStates):
            transition[i, j] = float(s[j])
        s = file.readline()
    
    model = MHMM_params()
    model.nbStates = nbStates
    model.dimension_gesture = dimension_gesture
    model.dimension_sound = dimension_sound
    model.nbMixtures = nbMixtures
    model.covarianceOffset = covarianceOffset
    model.EM_stopCriterion = EM_stopCriterion
    model.EM_maxSteps = EM_maxSteps
    model.EM_percentchg = EM_percentchg
    model.prior = prior
    model.transition = transition
    # model.likelihoodWindow = likelihoodWindow
    
    # Read GMM for each state
    model.states = []
    for i in range(nbStates):
        model.states.append(read_MultimodalGMM(file))
    
    return model

def read_PolyMultimodalHMM(file):
    # Read Number of models
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    nbModels  = int(s)
    
    # Read Play mode
    # ------------------------------
    s =  file.readline()
    while s[0] == '#':
        s =  file.readline()
    playMode  = int(s)
    
    # Read  Reference Model ==> Nowhere
    # ------------------------------
    read_MultimodalHMM(file)
    
    # Read Models
    # ------------------------------
    polyMHMM = []
    for i in range(nbModels):
        # Read model index
        # ------------------------------
        s =  file.readline()
        while s[0] == '#':
            s =  file.readline()
        # print "model index:", int(s)
        polyMHMM.append(read_MultimodalHMM(file))
    
    return polyMHMM

def plot_GMM(ellipses, nbGMM=1, ax_lim=[-1.5, 1.5, -1.2, 1.2]):
    fig = figure()
    rcParams['axes.color_cycle'] = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if nbGMM == 1:
        ax = fig.add_subplot(111, aspect='equal')
        colorIndex = 0
        for e in ellipses:
            if isinstance(e, list):
                e = e[0]
            e.set_facecolor(rcParams['axes.color_cycle'][colorIndex])
            ax.add_artist(e)
            colorIndex += 1
            colorIndex %=7
            
        ax.set_xlim(ax_lim[0], ax_lim[1])
        ax.set_ylim(ax_lim[2], ax_lim[3])
    else:
        nbCols = int(sqrt(nbGMM))
        for i in range(nbGMM):
            ax = fig.add_subplot(nbGMM/nbCols, nbCols, i+1, aspect='equal')
            colorIndex = 0
            for e in ellipses[i]:
                e.set_facecolor(rcParams['axes.color_cycle'][colorIndex])
                ax.add_artist(e)
                colorIndex += 1
                colorIndex %=7
            ax.set_xlim(ax_lim[0], ax_lim[1])
            ax.set_ylim(ax_lim[2], ax_lim[3])

def computeEllipse(mean, covariance, dim1, dim2, mixture=-1):
    nbMixtures = mean.shape[0]
    dim_total =  mean.shape[1]
    
    if mixture >= 0:
        mean2d = array([mean[mixture, dim1], mean[mixture, dim2]])
        cov2d = array([[covariance[mixture, dim1, dim1], covariance[mixture, dim1, dim2]],
                        [covariance[mixture, dim2, dim1], covariance[mixture, dim2, dim2]]])
        
        eigenVal, eigenVec = eig(cov2d)
        theta = arctan(real(eigenVec[1, 0]) / real(eigenVec[0, 0]))
        ell = Ellipse(xy=mean2d, width=2*sqrt(eigenVal[0]), height=2*sqrt(eigenVal[1]), angle=theta*180/pi)
        ell.set_alpha(0.5)
        
        return ell
    else:
        ells = []
        for i in range(nbMixtures):
            ells.append(computeEllipse(mean, covariance, dim1, dim2, i))
        return ells

def plotTransition(model, index=0):
    if not(isinstance(model, list)):
        model = [model]
    figure()
    imshow(model[index].transition, interpolation='nearest')


def main():
    # Read Concurrent Multimodal HMM models ==> List of MHMM classes
    f = open('/Users/francoise/Documents/Code/Python/__lib__/mhmm_lib/test/fileIO_PMHMM.txt', 'r')
    MHMM_model = read_PolyMultimodalHMM(f)
    plot_GMM(computeEllipse(MHMM_model[0].states[1].mean, MHMM_model[0].states[1].covariance, 0, 1))
    show()
    f.close()
    
    # # Read Multimodal HMM ==> MHMM class
    #     f = open('/Users/francoise/Documents/Code/Python/__lib__/mhmm_lib/test/fileIO_MHMM.txt', 'r')
    #     MHMM_model = read_MultimodalHMM(f)
    #     plot_GMM(MHMM_model.states[0].mean, MHMM_model.states[0].covariance, 0, 1)
    #     plot_GMM(MHMM_model.states[1].mean, MHMM_model.states[1].covariance, 0, 1)
    #     show()
    #     f.close()
    
    # # Read Concurrent Multimodal GMM models ==> List of MGMM classes
    #     f = open('/Users/francoise/Documents/Code/Python/__lib__/mhmm_lib/test/fileIO_PMGMM.txt', 'r')
    #     polyMGMM = read_PolyMultimodalGMM(f)
    #     plot_GMM(polyMGMM[1].mean, polyMGMM[1].covariance, 3, 7)
    #     f.close()
    
    # Read Multimodal HMM ==> MHMM class
    # f = open('/Users/francoise/Documents/Code/Python/__lib__/mhmm_lib/test/fileIO_MGMM.txt', 'r')
    #     MGMM_model = read_MultimodalGMM(f)
    #     plot_GMM(MGMM_model.mean, MGMM_model.covariance, 0, 2)
    #     f.close()

if __name__ == '__main__':
    main()
