//
// gmm.cpp
//
// Gaussian Mixture Model
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
//

#include "gmm.h"

#pragma mark > Constructors and Utilities
GMM::GMM(rtml_flags flags,
         TrainingSet *trainingSet,
         int nbMixtureComponents,
         float covarianceOffset)
: EMBasedLearningModel(flags, trainingSet)
{
    nbMixtureComponents_  = nbMixtureComponents;
    covarianceOffset_     = covarianceOffset;
    
    set_trainingSet(trainingSet);
    
    initTraining();
}


GMM::GMM(GMM const& src) : EMBasedLearningModel(src)
{
    _copy(this, src);
}

GMM& GMM::operator=(GMM const& src)
{
    if(this != &src)
    {
        _copy(this, src);
    }
    return *this;
};

GMM::~GMM()
{
    components.clear();
    mixtureCoeffs.clear();
    beta.clear();
}

#pragma mark > Accessors & Attributes
int GMM::get_nbMixtureComponents() const
{
    return nbMixtureComponents_;
}

float GMM::get_covarianceOffset() const
{
    return covarianceOffset_;
}

void GMM::set_nbMixtureComponents(int nbMixtureComponents)
{
    if (nbMixtureComponents < 1) throw invalid_argument("Number of mixture components must be > 0");
    if (nbMixtureComponents == nbMixtureComponents_) return;
    
    nbMixtureComponents_ = nbMixtureComponents;
    allocate();
    this->trained = false;
}

void GMM::set_covarianceOffset(float covarianceOffset)
{
    if (covarianceOffset <= 0.) throw invalid_argument("Covariance offset must be > 0");
    
    covarianceOffset_ = covarianceOffset;
    for (mixture_iterator component = components.begin() ; component != components.end(); ++component) {
        component->set_offset(covarianceOffset_);
    }
}

#pragma mark > Performance
void GMM::initPlaying()
{
    if (bimodal_)
        results.predicted_output.resize(dimension_ - dimension_input_);
}

double GMM::play(float *observation)
{
    double instantaneous_likelihood = likelihood(observation);
    if (bimodal_)
    {
        regression(observation, results.predicted_output);
        copy(results.predicted_output.begin(), results.predicted_output.end(), observation + dimension_input_);
    }
    return instantaneous_likelihood;
}

#pragma mark > Training
void GMM::initTraining()
{
    initParametersToDefault();
    initMeansWithFirstPhrase();
    updateInverseCovariances();
}

void GMM::finishTraining()
{
    LearningModel::finishTraining();
}

#pragma mark > JSON I/O
JSONNode GMM::to_json() const
{
    JSONNode json_gmm(JSON_NODE);
    json_gmm.set_name("GMM");
    
    // Write Parent: EM Learning Model
    JSONNode json_emmodel = EMBasedLearningModel::to_json();
    json_emmodel.set_name("EMBasedLearningModel");
    json_gmm.push_back(json_emmodel);
    
    // Scalar Attributes
    json_gmm.push_back(JSONNode("nbMixtureComponents", nbMixtureComponents_));
    json_gmm.push_back(JSONNode("covarianceOffset", covarianceOffset_));
    
    // Model Parameters
    json_gmm.push_back(vector2json(mixtureCoeffs, "mixtureCoefficients"));
    
    // Mixture Components
    JSONNode json_components(JSON_ARRAY);
    for (int c=0 ; c<nbMixtureComponents_ ; c++)
    {
        json_components.push_back(components[c].to_json());
    }
    json_components.set_name("components");
    json_gmm.push_back(json_components);
    
    return json_gmm;
}

void GMM::from_json(JSONNode root)
{
    try {
        assert(root.type() == JSON_NODE);
        JSONNode::iterator root_it = root.begin();
        
        // Get Parent: EMBasedLearningModel
        assert(root_it != root.end());
        assert(root_it->name() == "EMBasedLearningModel");
        assert(root_it->type() == JSON_NODE);
        EMBasedLearningModel::from_json(*root_it);
        ++root_it;
        
        // Get Mixture Components
        assert(root_it != root.end());
        assert(root_it->name() == "nbMixtureComponents");
        assert(root_it->type() == JSON_NUMBER);
        nbMixtureComponents_ = root_it->as_int();
        ++root_it;
        
        // Get Covariance Offset
        assert(root_it != root.end());
        assert(root_it->name() == "covarianceOffset");
        assert(root_it->type() == JSON_NUMBER);
        covarianceOffset_ = root_it->as_float();
        ++root_it;
        
        allocate();
        
        // Get Mixture Coefficients
        assert(root_it != root.end());
        assert(root_it->name() == "mixtureCoefficients");
        assert(root_it->type() == JSON_ARRAY);
        json2vector(*root_it, mixtureCoeffs, nbMixtureComponents_);
        ++root_it;
        
        // Get Gaussian Mixture Components
        assert(root_it != root.end());
        assert(root_it->name() == "components");
        assert(root_it->type() == JSON_ARRAY);
        for (int i=0 ; i<nbMixtureComponents_ ; i++) {
            components[i].from_json((*root_it)[i]);
        }
        
        updateInverseCovariances();
        
    } catch (JSONException &e) {
        throw JSONException(e);
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
    
    this->trained = true;
}

#pragma mark > Utilities
void GMM::_copy(GMM *dst, GMM const& src)
{
    EMBasedLearningModel::_copy(dst, src);
    dst->nbMixtureComponents_ = src.nbMixtureComponents_;
    dst->covarianceOffset_ = src.covarianceOffset_;
    dst->mixtureCoeffs = src.mixtureCoeffs;
    dst->components = src.components;
    
    dst->allocate();
}

void GMM::allocate()
{
    mixtureCoeffs.resize(nbMixtureComponents_);
    beta.resize(nbMixtureComponents_);
    components.assign(nbMixtureComponents_, GaussianDistribution(flags_, dimension_, dimension_input_, covarianceOffset_));
}

double GMM::obsProb(const float *observation, int mixtureComponent)
{
    double p(0.);
    
    if (mixtureComponent < 0) {
        for (mixtureComponent=0; mixtureComponent<nbMixtureComponents_; mixtureComponent++) {
            p += obsProb(observation, mixtureComponent);
        }
    } else {
        if (mixtureComponent >= nbMixtureComponents_)
            throw out_of_range("The index of the Gaussian Mixture Component is out of bounds");
        p = mixtureCoeffs[mixtureComponent] * components[mixtureComponent].likelihood(observation);
    }
    
    return p;
}

double GMM::obsProb_input(const float *observation_input, int mixtureComponent)
{
    if (!bimodal_)
        throw runtime_error("Model is not bimodal. Use the function 'obsProb'");
    double p(0.);
    
    if (mixtureComponent < 0) {
        for (mixtureComponent=0; mixtureComponent<nbMixtureComponents_; mixtureComponent++) {
            p += obsProb_input(observation_input, mixtureComponent);
        }
    } else {
        
        p = mixtureCoeffs[mixtureComponent] * components[mixtureComponent].likelihood_input(observation_input);
    }
    
    return p;
}

double GMM::obsProb_bimodal(const float *observation_input, const float *observation_output, int mixtureComponent)
{
    if (!bimodal_)
        throw runtime_error("Model is not bimodal. Use the function 'obsProb'");
    double p(0.);
    
    if (mixtureComponent < 0) {
        for (mixtureComponent=0; mixtureComponent<nbMixtureComponents_; mixtureComponent++) {
            p += obsProb_bimodal(observation_input, observation_output, mixtureComponent);
        }
    } else {
        
        p = mixtureCoeffs[mixtureComponent] * components[mixtureComponent].likelihood_bimodal(observation_input, observation_output);
    }
    
    return p;
}

#pragma mark > Training
void GMM::initMeansWithFirstPhrase()
{
    if (!this->trainingSet || this->trainingSet->is_empty())
        return;
    int step = this->trainingSet->begin()->second->length() / nbMixtureComponents_;
    
    int offset(0);
    for (int c=0; c<nbMixtureComponents_; c++) {
        for (int d=0; d<dimension_; d++) {
            components[c].mean[d] = 0.0;
        }
        for (int t=0; t<step; t++) {
            for (int d=0; d<dimension_; d++) {
                components[c].mean[d] += (*this->trainingSet->begin()->second)(offset+t, d) / float(step);
            }
        }
        offset += step;
    }
}

void GMM::setParametersToZero()
{
    for (int c=0; c<nbMixtureComponents_; c++) {
        mixtureCoeffs[c] = 0.;
        components[c].setParametersToZero(false);
    }
}

double GMM::train_EM_update()
{
    double log_prob(0.);
    
    int totalLength(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); ++it)
        totalLength += it->second->length();
    
    vector< vector<double> > p(nbMixtureComponents_);
    vector<double> E(nbMixtureComponents_, 0.0);
    for (int c=0; c<nbMixtureComponents_; c++) {
        p[c].resize(totalLength);
        E[c] = 0.;
    }
    
    int tbase(0);
    
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); ++it) {
        int T = it->second->length();
        for (int t=0; t<T; t++) {
            double norm_const(0.);
            for (int c=0; c<nbMixtureComponents_; c++)
            {
                if (bimodal_) {
                    p[c][tbase+t] = obsProb_bimodal(it->second->get_dataPointer_input(t),
                                                    it->second->get_dataPointer_output(t),
                                                    c);
                } else {
                    p[c][tbase+t] = obsProb(it->second->get_dataPointer(t), c);
                }
                
                if (p[c][tbase+t] == 0. || isnan(p[c][tbase+t]) || isinf(p[c][tbase+t])) {
                    p[c][tbase+t] = 1e-100;
                }
                norm_const += p[c][tbase+t];
            }
            for (int c=0; c<nbMixtureComponents_; c++) {
                p[c][tbase+t] /= norm_const;
                E[c] += p[c][tbase+t];
            }
            if (norm_const > 1.)
                cout << "Training Error: covarianceOffset is too small\n";//throw runtime_error("Training Error: covarianceOffset is too small");
            log_prob += log(norm_const);
        }
        tbase += T;
    }
    
    // Estimate Mixture coefficients
    for (int c=0; c<nbMixtureComponents_; c++) {
        mixtureCoeffs[c] = E[c]/double(totalLength);
    }
    
    // Estimate means
    for (int c=0; c<nbMixtureComponents_; c++) {
        for (int d=0; d<dimension_; d++) {
            components[c].mean[d] = 0.;
            tbase = 0;
            for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); ++it) {
                int T = it->second->length();
                for (int t=0; t<T; t++) {
                    components[c].mean[d] += p[c][tbase+t] * (*it->second)(t, d);
                }
                tbase += T;
            }
            components[c].mean[d] /= E[c];
        }
    }
    
    //estimate covariances
    for (int c=0; c<nbMixtureComponents_; c++) {
        for (int d1=0; d1<dimension_; d1++) {
            for (int d2=0; d2<dimension_; d2++) {
                components[c].covariance[d1 * dimension_ + d2] = 0.;
                tbase = 0;
                for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); ++it) {
                    int T = it->second->length();
                    for (int t=0; t<T; t++) {
                        components[c].covariance[d1 * dimension_ + d2] += p[c][tbase+t]
                        * ((*it->second)(t, d1) - components[c].mean[d1])
                        * ((*it->second)(t, d2) - components[c].mean[d2]);
                    }
                    tbase += T;
                }
                components[c].covariance[d1 * dimension_ + d2] /= E[c];
            }
        }
    }
    
    addCovarianceOffset();
    updateInverseCovariances();
    
    return log_prob;
}

void GMM::initParametersToDefault()
{
    double norm_coeffs(0.);
    for (int c=0; c<nbMixtureComponents_; c++) {
        for (int d=0; d<dimension_; d++) {
            for (int d2=0; d2<dimension_; d2++) {
                components[c].covariance[d * dimension_ + d2] = 1.;
            }
        }
        components[c].addOffset();
        mixtureCoeffs[c] = 1./float(nbMixtureComponents_);
        norm_coeffs += mixtureCoeffs[c];
    }
    for (int c=0; c<nbMixtureComponents_; c++) {
        mixtureCoeffs[c] /= norm_coeffs;
    }
}

void GMM::normalizeMixtureCoeffs()
{
    double norm_const(0.);
    for (int c=0; c<nbMixtureComponents_; c++) {
        norm_const += mixtureCoeffs[c];
    }
    if (norm_const > 0) {
        for (int c=0; c<nbMixtureComponents_; c++) {
            mixtureCoeffs[c] /= norm_const;
        }
    } else {
        for (int c=0; c<nbMixtureComponents_; c++) {
            mixtureCoeffs[c] = 1/float(nbMixtureComponents_);
        }
    }
}

void GMM::addCovarianceOffset()
{
    for (mixture_iterator component = components.begin() ; component != components.end(); ++component) {
        component->addOffset();
    }
}

void GMM::updateInverseCovariances()
{
    for (mixture_iterator component = components.begin() ; component != components.end() ; ++component)
        component->updateInverseCovariance();
}

#pragma mark > Performance
void GMM::regression(float *observation_input, vector<float>& predicted_output)
{
    int dimension_output = dimension_ - dimension_input_;
    predicted_output.assign(dimension_output, 0.0);
    vector<float> tmp_predicted_output(dimension_output, 0.0);
    
    for (int c=0; c<nbMixtureComponents_; c++) {
        components[c].regression(observation_input, tmp_predicted_output);
        for (int d = 0; d < dimension_output; ++d)
        {
            predicted_output[d] += beta[c] * tmp_predicted_output[d];
        }
    }
}

double GMM::likelihood(const float* observation, const float* observation_output)
{
    double likelihood(0.);
    for (int c=0; c<nbMixtureComponents_; c++) {
        if (bimodal_) {
            if (observation_output)
                beta[c] = obsProb_bimodal(observation, observation_output, c);
            else
                beta[c] = obsProb_input(observation, c);
        } else {
            beta[c] = obsProb(observation, c);
        }
        likelihood += beta[c];
    }
    for (int c=0; c<nbMixtureComponents_; c++) {
        beta[c] /= likelihood;
    }
    
    this->updateLikelihoodBuffer(likelihood);
    return likelihood;
}

#pragma mark > Deprecated: likeliestComponent
/*
int GMM::likeliestComponent()
{
    int component(0);
    double maxProb = mixtureCoeffs[component] / sqrt(components[component].covarianceDeterminant);
    for (int c=1 ; c<nbMixtureComponents; c++) {
        double prob = mixtureCoeffs[c] / sqrt(components[c].covarianceDeterminant);
        if (prob > maxProb) {
            component = c;
            maxProb = prob;
        }
    }
    return component;
}

int GMM::likeliestComponent(const float *obs)
{
    // !! here, the likeliest component is computed with the mixture coeffs => relevant ?
    int component(0);
    double maxProb = obsProb(obs, component);
    for (int c=1 ; c<nbMixtureComponents; c++) {
        double prob = obsProb(obs, c);
        if (prob > maxProb) {
            component = c;
            maxProb = prob;
        }
    }
    return component;
}
*/