/*
 * xmm.i
 *
 * Contact:
 * - Jules Francoise <jules.francoise@ircam.fr>
 *
 * This code has been initially authored by Jules Francoise
 * <http://julesfrancoise.com> during his PhD thesis, supervised by Frederic
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
    #include "xmmMatrix.hpp"
    #include "xmmCircularbuffer.hpp"
    #include "xmmJson.hpp"
    #include "xmmAttribute.hpp"
    #include "xmmEvents.hpp"
    #include "xmmGaussianDistribution.hpp"
    #include "xmmPhrase.hpp"
    #include "xmmTrainingSet.hpp"
    #include "xmmModelSharedParameters.hpp"
    #include "xmmModelParameters.hpp"
    #include "xmmModelResults.hpp"
    #include "xmmModelConfiguration.hpp"
    #include "xmmModelSingleClass.hpp"
    #include "xmmModel.hpp"
    #include "xmmKMeansParameters.hpp"
    #include "xmmKMeansResults.hpp"
    #include "xmmKMeans.hpp"
    #include "xmmGmmParameters.hpp"
    #include "xmmGmmSingleClass.hpp"
    #include "xmmGmm.hpp"
    #include "xmmHmmParameters.hpp"
    #include "xmmHmmResults.hpp"
    #include "xmmHmmSingleClass.hpp"
    #include "xmmHierarchicalHmm.hpp"
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
%include std_shared_ptr.i

//%apply (int DIM1, double* IN_ARRAY1) { (int dimension, double *observation) };
//%apply (int DIM1, double* ARGOUT_ARRAY1) { (double *modelLikelihoods, int dimension) };

namespace std {
    %template(vectord) vector<double>;
    %template(vectorf) vector<float>;
    %template(vectors) vector<string>;
    %template(sets) set<string>;
    %template(vectorgauss) vector<xmm::GaussianDistribution>;
    %template(vectorgmm) vector<xmm::SingleClassGMM>;
    %template(vectorhmm) vector<xmm::SingleClassHMM>;
    %template(mapgmm) map<string, xmm::SingleClassGMM>;
    %template(maphmm) map<string, xmm::SingleClassHMM>;
};

%include ../xmm_doc.i

%include "xmmMatrix.hpp"
%include "xmmCircularbuffer.hpp"
%include "xmmJson.hpp"
%include "xmmAttribute.hpp"

%template(Attribute_size_t) xmm::Attribute<std::size_t>;
%template(Attribute_int) xmm::Attribute<int>;
%template(Attribute_double) xmm::Attribute<double>;
%template(Attribute_float) xmm::Attribute<float>;
%template(Attribute_vectorString) xmm::Attribute<std::vector<std::string>>;

%include "xmmEvents.hpp"

%include "xmmGaussianDistribution.hpp"

%template(Attribute_CovarianceMode) xmm::Attribute<xmm::GaussianDistribution::CovarianceMode>;

%include "xmmPhrase.hpp"
%include "xmmTrainingSet.hpp"

%shared_ptr(xmm::Writable)
%shared_ptr(xmm::SharedParameters)
%include "xmmModelSharedParameters.hpp"

%include "xmmModelParameters.hpp"
%include "xmmModelResults.hpp"
%include "xmmModelConfiguration.hpp"

%include "xmmModelSingleClass.hpp"

%template(TrainingEventGenerator) xmm::EventGenerator< xmm::TrainingEvent >;

%include "xmmModel.hpp"

%include "xmmKMeansParameters.hpp"

%template(ClassParametersKMeans) xmm::ClassParameters< xmm::KMeans >;

%include "xmmKMeansResults.hpp"

%template(ResultsKMeans) xmm::Results< xmm::KMeans >;

%include "xmmKMeans.hpp"

%include "xmmGmmParameters.hpp"
%template(ClassParametersGMM) xmm::ClassParameters< xmm::GMM >;
%template(ClassResultsGMM) xmm::ClassResults< xmm::GMM >;
%template(ResultsGMM) xmm::Results< xmm::GMM >;
%template(ConfigurationGMM) xmm::Configuration< xmm::GMM >;

%include "xmmGmmSingleClass.hpp"

%template(Model_SingleClassGMM_GMM_) xmm::Model< xmm::SingleClassGMM, xmm::GMM >;

%include "xmmGmm.hpp"

%include "xmmHmmParameters.hpp"
%template(Attribute_TransitionMode) xmm::Attribute<xmm::HMM::TransitionMode>;
%template(Attribute_RegressionEstimator) xmm::Attribute<xmm::HMM::RegressionEstimator>;

%include "xmmHmmResults.hpp"
%template(ClassParametersHMM) xmm::ClassParameters< xmm::HMM >;
%template(ClassResultsHMM) xmm::ClassResults< xmm::HMM >;
%template(ResultsHMM) xmm::Results< xmm::HMM >;
%template(ConfigurationHMM) xmm::Configuration< xmm::HMM >;
%include "xmmHmmSingleClass.hpp"

%template(Model_SingleClassHMM_HMM_) xmm::Model< xmm::SingleClassHMM, xmm::HMM >;

%include "xmmHierarchicalHmm.hpp"

%extend xmm::Configuration<xmm::GMM> {
    xmm::ClassParameters<xmm::GMM>& __getitem__(std::string label) {
        return (*($self))[label];
    }
}
%extend xmm::Configuration<xmm::HMM> {
    xmm::ClassParameters<xmm::HMM>& __getitem__(std::string label) {
        return (*($self))[label];
    }
}
