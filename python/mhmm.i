%module(docstring="Multimodal Hidden Markov Models Library") mhmm

%{
	#define SWIG_FILE_WITH_INIT
	// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
	#include "phrase.h"
	#include "gesturesound_phrase.h"
	#include "gesture_phrase.h"
	#include "notifiable.h"
	#include "training_set.h"
	#include "learning_model.h"
	#include "em_based_learning_model.h"
	#include "concurrent_models.h"
	#include "multimodal_gmm.h"
	#include "concurrent_mgmm.h"
	#include "multimodal_hmm.h"
	#include "concurrent_mhmm.h"
	#include "hierarchical_model.h"
	#include "Hierarchical_mhmm_submodel.h"
	#include "hierarchical_mhmm.h"
	#include "gmm.h"
	#include "concurrent_gmm.h"
	#include "hmm.h"
	#include "concurrent_hmm.h"
	#include "Hierarchical_hmm_submodel.h"
	#include "hierarchical_hmm.h"
	#include <fstream>
	#include <sstream>
%}

%exception {
    try {
        $action
    }
    catch (exception const& e) {
        PyErr_SetString(PyExc_IndexError,e.what());
        SWIG_fail;
    }
}

%include numpy.i
%include std_string.i
%include std_map.i
%include mhmm_doc.i
%include "phrase.h"
%include "gesturesound_phrase.h"
%include "gesture_phrase.h"
%include "notifiable.h"
%include "training_set.h"
%include "learning_model.h"
%include "em_based_learning_model.h"
%include "concurrent_models.h"
%include "multimodal_gmm.h"
%include "concurrent_mgmm.h"
%include "multimodal_hmm.h"
%include "concurrent_mhmm.h"
%include "hierarchical_model.h"
%include "Hierarchical_mhmm_submodel.h"
%include "hierarchical_mhmm.h"
%include "gmm.h"
%include "concurrent_gmm.h"
%include "hmm.h"
%include "concurrent_hmm.h"
%include "Hierarchical_hmm_submodel.h"
%include "hierarchical_hmm.h"

%init %{
	import_array();
%}

// PHRASE
// ====================================
%apply (int DIM1, double* IN_ARRAY1) { (int dimension_total, double *observation) }; // TODO: Convert to float ? Possible with numpy ?
%template(uPhrase1) Phrase<true, 1>;
%template(uPhrase2) Phrase<true, 2>;
%template(mPhrase)  GestureSoundPhrase<true>;

// TRAINING SET
// ====================================
%template(_uTrainingSetBase1) _TrainingSetBase<Phrase<true, 1>, int>;
%template(_uTrainingSetBase2) _TrainingSetBase<Phrase<true, 2>, int>;
%template(_mTrainingSetBase)  _TrainingSetBase<GestureSoundPhrase<true>, int>;
%template(_gTrainingSetBase)  _TrainingSetBase<GesturePhrase<true>, int>;
%template(uTrainingSet1) TrainingSet<Phrase<true, 1>, int>;
%template(uTrainingSet2) TrainingSet<Phrase<true, 2>, int>;
%template(mTrainingSet)  TrainingSet<GestureSoundPhrase<true>, int>;

// LEARNING MODEL: BASE DEFINITIONS
// ====================================
%template(uLearningModel1) LearningModel<Phrase<true, 1>, int>;
%template(uLearningModel2) LearningModel<Phrase<true, 2>, int>;
%template(mLearningModel)  LearningModel<GestureSoundPhrase<true>, int>;

%template(uEMBasedLearningModel1) EMBasedLearningModel<Phrase<true, 1>, int>;
%template(uEMBasedLearningModel2) EMBasedLearningModel<Phrase<true, 2>, int>;
%template(mEMBasedLearningModel)  EMBasedLearningModel<GestureSoundPhrase<true>, int>;

// MULTIMODAL GMM & MULTIMODAL HMM
// ====================================
%apply (int DIM1, double* IN_ARRAY1) { (int dimension_gesture_, double *observation_gesture) };
%apply (int DIM1, double* ARGOUT_ARRAY1) { (int dimension_sound_, double *observation_sound_out), (int nbModels_, double *likelihoods_) };
%apply (int DIM1, double* ARGOUT_ARRAY1) { (int nbMixtureComponents_, double *beta_), (int nbStates_, double *alpha_), (int dimension_sound_square, double *outCovariance) };
%template(MGMM) MultimodalGMM<true>;
%template(MHMM) MultimodalHMM<true>;
%apply (int DIM1, double* IN_ARRAY1) { (int dimension_, double *observation) };
%template(pyGMM) GMM<true>;
%template(pyHMM) HMM<true>;

%template(trainMap) std::map<int, int>;

%apply (int DIM1, double* ARGOUT_ARRAY1) { (int nbModels_, double *likelihoods), (int nbModels__, double *cumulativelikelihoods) };
%template(_mgmmConcurrentModels)  ConcurrentModels<MultimodalGMM<true>, GestureSoundPhrase<true>, int>;
%template(_mhmmConcurrentModels)  ConcurrentModels<MultimodalHMM<true>, GestureSoundPhrase<true>, int>;
%template(_gmmConcurrentModels)   ConcurrentModels<GMM<true>, Phrase<true, 1>, int>;
%template(_hmmConcurrentModels)   ConcurrentModels<HMM<true>, Phrase<true, 1>, int>;

%template(PolyMGMM) ConcurrentMGMM<true>;
%template(PolyMHMM) ConcurrentMHMM<true>;
%template(PolyGMM)  ConcurrentGMM<true>;
%template(PolyHMM)  ConcurrentHMM<true>;

// HIERARCHICAL MHMM
// ====================================
%apply (int DIM1, double* IN_ARRAY1) { (int nbPrimitives, double *prior_), (int nbPrimitivesSquared, double *trans_) };

%template(_HMHMMSubmodel) HierarchicalMHMMSubmodel<true>;
%template(_hmhmmConcurrentModels)  ConcurrentModels<HierarchicalMHMMSubmodel<true>, GestureSoundPhrase<true>, int>;
%template(_mhmmHierarchicalModel)  HierarchicalModel<HierarchicalMHMMSubmodel<true>, GestureSoundPhrase<true>, int>;
%template(HMHMM) HierarchicalMHMM<true>;

%template(_HHMMSubmodel) HierarchicalHMMSubmodel<true>;
%template(_hhmmConcurrentModels)  ConcurrentModels<HierarchicalHMMSubmodel<true>, Phrase<true, 1>, int>;
%template(_hmmHierarchicalModel)  HierarchicalModel<HierarchicalHMMSubmodel<true>, Phrase<true, 1>, int>;
%template(HHMM) HierarchicalHMM<true>;
