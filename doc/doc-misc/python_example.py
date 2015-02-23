import numpy as np
import mhmm

# Load Training Data
training_motion_1 = np.genfromtxt('training_motion_1.txt')
training_motion_2 = np.genfromtxt('training_motion_2.txt')
training_sound_1 = np.genfromtxt('training_sound_1.txt')
training_sound_2 = np.genfromtxt('training_sound_2.txt')

dim_gesture = training_motion_1.shape[1]
dim_sound = training_sound_1.shape[1]

# Create a multimodal training set
training_set = mhmm.TrainingSet(mhmm.BIMODAL)
training_set.set_dimension(dim_gesture + dim_sound)
training_set.set_dimension_input(dim_sound)

# Record First Phrase
for frame_motion, frame_sound in zip(training_motion_1, training_sound_1):
    training_set.recordPhrase_input (0, frame_motion)
    training_set.recordPhrase_output(0, frame_sound)
training_set.setPhraseLabel(0, mhmm.Label('one'))

# Record Second Phrase
for frame_motion, frame_sound in zip(training_motion_2, training_sound_2):
    training_set.recordPhrase_input (1, frame_motion)
    training_set.recordPhrase_output(1, frame_sound)
training_set.setPhraseLabel(1, mhmm.Label('two'))

# Instantiate and Train a Hierarchical Multimodal HMM
xmm = mhmm.HierarchicalHMM(mhmm.BIMODAL, training_set)
xmm.set_nbStates(10)
xmm.set_nbMixtureComponents(1)
xmm.set_varianceOffset(0.1, 0.01)
xmm.train()

# Perform joint recognition and Mapping
test_motion = np.genfromtxt('test_motion.txt')
predicted_sound = np.zeros((len(test_motion), dim_sound))
log_likelihoods = np.zeros((len(test_motion), xmm.size()))
xmm.performance_init()
for t, frame_motion in enumerate(test_motion):
    xmm.performance_update(frame)
    predicted_sound[t, :] = xmm.results_predicted_output
    log_likelihoods[t, :] = xmm.results_log_likelihoods


