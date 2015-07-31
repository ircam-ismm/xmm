/*
 * xmm.i
 *
 * Contact:
 * - Jules Françoise <jules.francoise@ircam.fr>
 *
 * This code has been initially authored by Jules Françoise
 * <http://julesfrancoise.com> during his PhD thesis, supervised by Frédéric
 * Bevilacqua <href="http://frederic-bevilacqua.net>, in the Sound Music
 * Movement Interaction team <http://ismm.ircam.fr> of the
 * STMS Lab - IRCAM, CNRS, UPMC (2011-2015).
 *
 * Copyright (C) 2015 UPMC, Ircam-Centre Pompidou.
 *
 * This File is part of XMM.
 *
 * XMM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * XMM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with XMM.  If not, see <http://www.gnu.org/licenses/>.
 */

%module(docstring="XMM - Probabilistic Models for Motion Recognition and Mapping") xmm

#pragma SWIG nowarn=362,503

%{
    #define SWIG_FILE_WITH_INIT
    #include <fstream>
    #include <sstream>
	#include "xmm_common.h"
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
	#include "kmeans.h"
%}

%init %{
    import_array();
%}

%exception {
    try {
        $action
    }
    catch (std::exception const& e) {
        PyErr_SetString(PyExc_IndexError,e.what());
        SWIG_fail;
    }
}

%include std_vector.i
%include numpy.i
%include std_string.i
%include std_map.i
%include std_set.i

%apply (int DIM1, double* IN_ARRAY1) { (int dimension, double *observation) };
%apply (int DIM1, double* ARGOUT_ARRAY1) { (double *modelLikelihoods, int dimension) };

namespace std {
    %template(vectord) vector<double>;
    %template(vectorf) vector<float>;
    %template(vectors) vector<string>;
    %template(vectorl) vector<xmm::Label>;
    %template(setl) set<xmm::Label>;
    %template(vectorgauss) vector<xmm::GaussianDistribution>;
    %template(vectorgmm) vector<xmm::GMM>;
    %template(vectorhmm) vector<xmm::HMM>;
    %template(mapgmm) map<xmm::Label, xmm::GMM>;
    %template(maphmm) map<xmm::Label, xmm::HMM>;
};

%include ../xmm_doc.i
%include "xmm_common.h"
%include "phrase.h"
%include "label.h"
%include "training_set.h"
%include "probabilistic_model.h"
%include "gaussian_distribution.h"
%include "kmeans.h"
%include "gmm.h"
%include "hmm.h"
%include "model_group.h"

%template(_MODELGROUP_GMM) xmm::ModelGroup<xmm::GMM>;
%template(_MODELGROUP_HMM) xmm::ModelGroup<xmm::HMM>;

%include "gmm_group.h"
%include "hierarchical_hmm.h"
