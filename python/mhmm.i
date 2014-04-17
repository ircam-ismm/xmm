%module(docstring="Multimodal Hidden Markov Models Library") mhmm

%{
	#define SWIG_FILE_WITH_INIT
	// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <fstream>
    #include <sstream>
	#include "mbd_common.h"
    #include "phrase.h"
    #include "label.h"
	#include "training_set.h"
	#include "probabilistic_model.h"
	#include "model_group.h"
    #include "gaussian_distribution.h"
	#include "gmm.h"
	#include "gmm_group.h"
	#include "hmm.h"
	#include "hierarchical_hmm.h"
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

%include "std_vector.i"
%include numpy.i
%include std_string.i
%include std_map.i

%apply (int DIM1, double* IN_ARRAY1) { (int dimension, double *observation) };
%apply (int DIM1, double* ARGOUT_ARRAY1) { (double *modelLikelihoods, int dimension) };

namespace std {
   %template(vectord) vector<double>;
   %template(vectorf) vector<float>;
   %template(vectorgauss) vector<GaussianDistribution>;
   %template(vectorgmm) vector<GMM>;
   %template(vectorhmm) vector<HMM>;
};

%init %{
    import_array();
%}

// %typemap(out) vectord {
//     int length = $1.size();
//     $result = PyArray_FromDims(1, &amp;length, PyArray_DOUBLE);
//     memcpy(PyArray_DATA($result),&amp;((*(&amp;$1))[0]),sizeof(double)*length);
// }

// %include mhmm_doc.i
%include "mbd_common.h"
%include "phrase.h"
%include "label.h"
%include "training_set.h"
%include "probabilistic_model.h"
%include "gaussian_distribution.h"
%include "gmm.h"
%include "hmm.h"
%include "model_group.h"

%template(_MODELGROUP_GMM) ModelGroup<GMM>;
%template(_MODELGROUP_HMM) ModelGroup<HMM>;

%include "gmm_group.h"
%include "hierarchical_hmm.h"
