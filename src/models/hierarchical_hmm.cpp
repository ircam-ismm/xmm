/*
 * hierarchical_hmm.cpp
 *
 * Hierarchical Hidden Markov Model for continuous recognition and regression
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

#include "hierarchical_hmm.h"

#pragma mark -
#pragma mark Constructors
HierarchicalHMM::HierarchicalHMM(rtml_flags flags,
                                 TrainingSet *_globalTrainingSet,
                                 GaussianDistribution::COVARIANCE_MODE covariance_mode)
: ModelGroup< HMM >(flags|HIERARCHICAL, _globalTrainingSet),
  incrementalLearning_(HHMM_DEFAULT_INCREMENTALLEARNING),
  forwardInitialized_(false)
{
    set_covariance_mode(covariance_mode);
}

HierarchicalHMM::~HierarchicalHMM()
{
    prior.clear();
    transition.clear();
    exitTransition.clear();
    V1_.clear();
    V2_.clear();
}

void HierarchicalHMM::clear()
{
    ModelGroup<HMM>::clear();
    prior.clear();
    transition.clear();
    exitTransition.clear();
}

#pragma mark -
#pragma mark Get & Set
int HierarchicalHMM::get_nbStates() const
{
    return this->referenceModel_.get_nbStates();
}

void HierarchicalHMM::set_nbStates(int nbStates_)
{
    PREVENT_ATTR_CHANGE();
    this->referenceModel_.set_nbStates(nbStates_);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_nbStates(nbStates_);
    }
}

int HierarchicalHMM::get_nbMixtureComponents() const
{
    return this->referenceModel_.get_nbMixtureComponents();
}

void HierarchicalHMM::set_nbMixtureComponents(int nbMixtureComponents_)
{
    PREVENT_ATTR_CHANGE();
    this->referenceModel_.set_nbMixtureComponents(nbMixtureComponents_);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_nbMixtureComponents(nbMixtureComponents_);
    }
}

double HierarchicalHMM::get_varianceOffset_relative() const
{
    return this->referenceModel_.get_varianceOffset_relative();
}

double HierarchicalHMM::get_varianceOffset_absolute() const
{
    return this->referenceModel_.get_varianceOffset_absolute();
}

void HierarchicalHMM::set_varianceOffset(double varianceOffset_relative, double varianceOffset_absolute)
{
    PREVENT_ATTR_CHANGE();
    this->referenceModel_.set_varianceOffset(varianceOffset_relative, varianceOffset_absolute);
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.set_varianceOffset(varianceOffset_relative, varianceOffset_absolute);
    }
}

GaussianDistribution::COVARIANCE_MODE HierarchicalHMM::get_covariance_mode() const
{
    return this->referenceModel_.get_covariance_mode();
}

void HierarchicalHMM::set_covariance_mode(GaussianDistribution::COVARIANCE_MODE covariance_mode)
{
    PREVENT_ATTR_CHANGE();
    if (covariance_mode == get_covariance_mode()) return;
    try {
        this->referenceModel_.set_covariance_mode(covariance_mode);
    } catch (exception& e) {
        if (strncmp(e.what(), "Non-invertible matrix", 21) != 0)
            throw runtime_error(e.what());
    }
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.set_covariance_mode(covariance_mode);
    }
}

REGRESSION_ESTIMATOR HierarchicalHMM::get_regression_estimator() const
{
    return this->referenceModel_.get_regression_estimator();
}

void HierarchicalHMM::set_regression_estimator(REGRESSION_ESTIMATOR regression_estimator)
{
    this->referenceModel_.set_regression_estimator(regression_estimator);
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.set_regression_estimator(regression_estimator);
    }
}

bool HierarchicalHMM::get_estimateMeans() const
{
    return this->referenceModel_.estimateMeans_;
}

void HierarchicalHMM::set_estimateMeans(bool _estimateMeans)
{
    PREVENT_ATTR_CHANGE();
    this->referenceModel_.estimateMeans_ = _estimateMeans;
    for (model_iterator it = this->models.begin() ; it != this->models.end() ; it++)
        it->second.estimateMeans_ = _estimateMeans;
}

string HierarchicalHMM::get_transitionMode() const
{
    return this->referenceModel_.get_transitionMode();
}

void HierarchicalHMM::set_transitionMode(string transMode_str)
{
    PREVENT_ATTR_CHANGE();
    this->referenceModel_.set_transitionMode(transMode_str);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_transitionMode(transMode_str);
    }
}

void HierarchicalHMM::addExitPoint(int state, float proba)
{
    PREVENT_ATTR_CHANGE();
    this->referenceModel_.addExitPoint(state, proba);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.addExitPoint(state, proba);
    }
}


string HierarchicalHMM::get_learningMode() const
{
    string learningMode;
    if (incrementalLearning_) {
        learningMode = "incremental";
    } else {
        learningMode = "ergodic";
    }
    return learningMode;
}


void HierarchicalHMM::set_learningMode(string learningMode)
{
    PREVENT_ATTR_CHANGE();
    if (learningMode == "incremental") {
        incrementalLearning_ = true;
    } else if (learningMode == "ergodic") {
        incrementalLearning_ = false;
    } else {
        throw invalid_argument("'learningMode' should be 'incremental' or 'ergodic'");
    }
}


void HierarchicalHMM::get_prior(vector<double>& prior) const
{
    prior.resize(this->size());
    int l(0);
    for (const_model_iterator it = this->models.begin(); it != this->models.end(); ++it) {
        prior[l++] = this->prior.at(it->first);
    }
}


void HierarchicalHMM::set_prior(vector<double> const& prior)
{
    PREVENT_ATTR_CHANGE();
    try {
        int l(0);
        for (model_iterator it = this->models.begin() ; it != this->models.end() ; ++it) {
            this->prior[it->first] = max(prior[l++], 0.0);
        }
        this->normalizeTransitions();
    } catch (exception &e) {
        throw invalid_argument("Wrong format for prior");
    }
}


void HierarchicalHMM::get_transition(vector<double>& trans) const
{
    unsigned int nbPrimitives = this->size();
    trans.resize(nbPrimitives*nbPrimitives);
    int l(0);
    
    for (const_model_iterator srcit = this->models.begin(); srcit != this->models.end(); ++srcit) {
        for (const_model_iterator dstit = this->models.begin(); dstit != this->models.end(); ++dstit) {
            trans[l++] = this->transition.at(srcit->first).at(dstit->first);
        }
    };
}


void HierarchicalHMM::set_transition(vector<double> const& trans)
{
    PREVENT_ATTR_CHANGE();
    try {
        int l(0);
        for (model_iterator srcit = this->models.begin(); srcit != this->models.end(); ++srcit) {
            for (model_iterator dstit = this->models.begin(); dstit != this->models.end(); ++dstit) {
                this->transition[srcit->first][dstit->first] = max(trans[l++], 0.0);
            }
        }
        this->normalizeTransitions();
    } catch (exception &e) {
        throw invalid_argument("Wrong format for transition");
    }
}


void HierarchicalHMM::get_exitTransition(vector<double>& exittrans) const
{
    exittrans.resize(this->size());
    int l(0);
    
    for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it)
    {
        exittrans[l++] = this->exitTransition.at(it->first);
    }
}


void HierarchicalHMM::set_exitTransition(vector<double> const& exittrans)
{
    PREVENT_ATTR_CHANGE();
    try {
        int l(0);
        for (model_iterator it = this->models.begin() ; it != this->models.end() ; ++it) {
            this->exitTransition[it->first] = max(exittrans[l++], 0.0);
        }
        this->normalizeTransitions();
    } catch (exception &e) {
        throw invalid_argument("Wrong format for prior");
    }
}


void HierarchicalHMM::normalizeTransitions()
{
    double sumPrior(0.0);
    for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; ++srcit)
    {
        sumPrior += prior[srcit->first];
        double sumTrans(0.0);
        for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; ++dstit)
            sumTrans += transition[srcit->first][dstit->first];
        if (sumTrans > 0.0)
            for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; ++dstit)
                transition[srcit->first][dstit->first] /= sumTrans;
    }
    for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; ++srcit)
        prior[srcit->first] /= sumPrior;
}


void HierarchicalHMM::setOneTransition(Label srcSegmentLabel, Label dstSegmentLabel, double proba)
{
    PREVENT_ATTR_CHANGE();
    transition[srcSegmentLabel][dstSegmentLabel] = min(proba, 1.);
    normalizeTransitions();
}

#pragma mark -
#pragma mark High level parameters: update and estimation


void HierarchicalHMM::updateTransitionParameters()
{
    if (this->size() == prior.size()) // number of primitives has not changed
        return;
    
    if (incrementalLearning_) {          // incremental learning: use regularization to preserve transition
        updatePrior_incremental();
        updateTransition_incremental();
    } else {                            // ergodic learning: set ergodic prior and transition probabilities
        updatePrior_ergodic();
        updateTransition_ergodic();
    }
    
    updateExitProbabilities(); // Update exit probabilities of Submodels (signal level)
}


void HierarchicalHMM::updatePrior_incremental()
{
    unsigned long oldNbPrim = prior.size();
    unsigned long nbPrimitives = this->size();
    
    if (oldNbPrim > 0)
    {
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it)
            if (prior.find(it->first) == prior.end())
            {
                prior[it->first] += double(HHMM_DEFAULT_REGULARIZATIONFACTOR);
                prior[it->first] /= double(nbPrimitives + HHMM_DEFAULT_REGULARIZATIONFACTOR) ;
            } else {
                prior[it->first] = 1. / double(nbPrimitives + HHMM_DEFAULT_REGULARIZATIONFACTOR);
            }
    } else {
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it)
            prior[it->first] = 1. / double(nbPrimitives);
    }
    
}


void HierarchicalHMM::updateTransition_incremental()
{
    unsigned long oldNbPrim = prior.size();
    unsigned long nbPrimitives = this->size();
    
    if (oldNbPrim > 0)
    {
        for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; ++srcit)
        {
            for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; ++dstit)
            {
                if (transition.find(srcit->first) == transition.end() || transition[srcit->first].find(dstit->first) == transition[srcit->first].end())
                {
                    transition[srcit->first][dstit->first] = 1/double(nbPrimitives+HHMM_DEFAULT_REGULARIZATIONFACTOR);
                } else {
                    transition[srcit->first][dstit->first] += double(HHMM_DEFAULT_REGULARIZATIONFACTOR);
                    transition[srcit->first][dstit->first] /= double(nbPrimitives+HHMM_DEFAULT_REGULARIZATIONFACTOR);
                }
            }
            
            if (exitTransition.find(srcit->first) == exitTransition.end())
            {
                exitTransition[srcit->first] = 1/double(nbPrimitives+HHMM_DEFAULT_REGULARIZATIONFACTOR);
            } else {
                exitTransition[srcit->first] += double(HHMM_DEFAULT_REGULARIZATIONFACTOR);
                exitTransition[srcit->first] /= double(nbPrimitives+HHMM_DEFAULT_REGULARIZATIONFACTOR);
            }
        }
    } else {
        for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; ++srcit)
        {
            exitTransition[srcit->first] = HHMM_DEFAULT_EXITTRANSITION;
            
            for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; ++dstit)
                transition[srcit->first][dstit->first] = 1/(double)nbPrimitives;
        }
    }
}


void HierarchicalHMM::updatePrior_ergodic()
{
    int nbPrimitives = this->size();
    for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it)
        prior[it->first] = 1/double(nbPrimitives);
}


void HierarchicalHMM::updateTransition_ergodic()
{
    int nbPrimitives = this->size();
    for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; ++srcit)
    {
        exitTransition[srcit->first] = HHMM_DEFAULT_EXITTRANSITION;
        for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; ++dstit)
            transition[srcit->first][dstit->first] =  1/double(nbPrimitives);
    }
}


void HierarchicalHMM::updateExitProbabilities()
{
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.updateExitProbabilities();
    }
}

void HierarchicalHMM::updateTrainingSet(Label const& label)
{
    ModelGroup<HMM>::updateTrainingSet(label);
    updateTransitionParameters();
}

#pragma mark -
#pragma mark Forward Algorithm
void HierarchicalHMM::forward_init(vector<float> const& observation)
{
    double norm_const(0.0) ;
    
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++)
    {
        int N = it->second.get_nbStates();
        
        for (int i=0; i<3; i++) {
            it->second.alpha_h[i].assign(N, 0.0);
        }
        
        // Compute Emission probability and initialize on the first state of the primitive
        it->second.alpha_h[0][0] = this->prior[it->first];
        if (bimodal_) {
            it->second.alpha_h[0][0] *= it->second.states_[0].obsProb_input(&observation[0]);
        } else {
            it->second.alpha_h[0][0] *= it->second.states_[0].obsProb(&observation[0]);
        }
        it->second.results_instant_likelihood = it->second.alpha_h[0][0] ;
        it->second.updateLikelihoodBuffer(it->second.results_instant_likelihood);
        norm_const += it->second.alpha_h[0][0] ;
    }
    
    // Normalize Alpha variables
    for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
        int N = it->second.get_nbStates();
        for (int e=0 ; e<3 ; e++)
            for (int k=0 ; k<N ; k++)
                it->second.alpha_h[e][k] /= norm_const;
    }
    
    forwardInitialized_ = true;
}

void HierarchicalHMM::forward_update(vector<float> const& observation)
{
    double norm_const(0.0) ;
    
    // Frontier Algorithm: variables
    double tmp(0);
    vector<double> front; // frontier variable : intermediate computation variable
    
    // Intermediate variables: compute the sum of probabilities of making a transition to a new primitive
    likelihoodAlpha(1, V1_);
    likelihoodAlpha(2, V2_);
    
    // FORWARD UPDATE
    // --------------------------------------
    for (model_iterator dstit = this->models.begin(); dstit != this->models.end(); dstit++) {
        int N = dstit->second.get_nbStates();
        
        // 1) COMPUTE FRONTIER VARIABLE
        //    --------------------------------------
        front.resize(N) ;
        
        // k=0: first state of the primitive
        front[0] = dstit->second.transition_[0] * dstit->second.alpha_h[0][0] ;
        
        int i(0);
        for (model_iterator srcit = this->models.begin(); srcit != this->models.end(); srcit++, i++) {
            front[0] += V1_[i] * this->transition[srcit->first][dstit->first] + this->prior[dstit->first] * V2_[i];
        }
        
        // k>0: rest of the primitive
        for (int k=1 ; k<N ; ++k)
        {
            front[k] = 0;
            for (int j = 0 ; j < N ; ++j)
            {
                front[k] += dstit->second.transition_[j*N+k] / (1 - dstit->second.exitProbabilities_[j]) * dstit->second.alpha_h[0][j] ;
            }
        }
        
        for (int i=0; i<3; i++) {
            for (int k=0; k<N; k++){
                dstit->second.alpha_h[i][k] = 0.0;
            }
        }
        
        // 2) UPDATE FORWARD VARIABLE
        //    --------------------------------------
        
        dstit->second.results_exit_likelihood = 0.0;
        dstit->second.results_instant_likelihood = 0.0;
        
        // end of the primitive: handle exit states
        for (int k=0 ; k<N ; ++k)
        {
            if (bimodal_)
                tmp = dstit->second.states_[k].obsProb_input(&observation[0]) * front[k];
            else
                tmp = dstit->second.states_[k].obsProb(&observation[0]) * front[k];
            
            dstit->second.alpha_h[2][k] = this->exitTransition[dstit->first] * dstit->second.exitProbabilities_[k] * tmp ;
            dstit->second.alpha_h[1][k] = (1 - this->exitTransition[dstit->first]) * dstit->second.exitProbabilities_[k] * tmp ;
            dstit->second.alpha_h[0][k] = (1 - dstit->second.exitProbabilities_[k]) * tmp;
            
            dstit->second.results_exit_likelihood += dstit->second.alpha_h[1][k] +dstit->second.alpha_h[2][k];
            dstit->second.results_instant_likelihood += dstit->second.alpha_h[0][k] + dstit->second.alpha_h[1][k] + dstit->second.alpha_h[2][k];
            
            norm_const += tmp;
        }
        
        dstit->second.updateLikelihoodBuffer(dstit->second.results_instant_likelihood);
        dstit->second.results_exit_ratio = dstit->second.results_exit_likelihood / dstit->second.results_instant_likelihood;
    }
    
    // Normalize Alpha variables
    for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
        int N = it->second.get_nbStates();
        for (int e=0 ; e<3 ; e++)
            for (int k=0 ; k<N ; k++)
                it->second.alpha_h[e][k] /= norm_const;
    }
}


void HierarchicalHMM::likelihoodAlpha(int exitNum, vector<double> &likelihoodVector) const
{
    if (exitNum<0) { // Likelihood over all exit states
        int l(0);
        for (const_model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            likelihoodVector[l] = 0.0;
            for (int exit = 0; exit<3; ++exit) {
                for (int k=0; k<it->second.get_nbStates(); k++){
                    likelihoodVector[l] += it->second.alpha_h[exit][k];
                }
            }
            l++;
        }
        
    } else { // Likelihood for exit state "exitNum"
        int l(0);
        for (const_model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            likelihoodVector[l] = 0.0;
            for (int k=0; k<it->second.get_nbStates(); k++){
                likelihoodVector[l] += it->second.alpha_h[exitNum][k];
            }
            l++;
        }
    }
}


void HierarchicalHMM::remove(Label const& label)
{
    ModelGroup<HMM>::remove(label);
    updateTransitionParameters();
}

#pragma mark -
#pragma mark Playing
void HierarchicalHMM::performance_init()
{
    ModelGroup<HMM>::performance_init();
    V1_.resize(this->size()) ;
    V2_.resize(this->size()) ;
    forwardInitialized_ = false;
}

void HierarchicalHMM::performance_update(vector<float> const& observation)
{
    if (forwardInitialized_) {
        this->forward_update(observation);
    } else {
        this->forward_init(observation);
    }
    
    update_likelihood_results();
    
    // Compute time progression
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.updateAlphaWindow();
        it->second.updateTimeProgression();
    }
    
    if (bimodal_) {
        unsigned int dimension = this->referenceModel_.dimension();
        unsigned int dimension_input = this->referenceModel_.dimension_input();
        unsigned int dimension_output = dimension - dimension_input;
        
        for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
            it->second.regression(observation, it->second.results_predicted_output);
        }
        
        if (this->performanceMode_ == this->LIKELIEST) {
            copy(this->models[results_likeliest].results_predicted_output.begin(),
                 this->models[results_likeliest].results_predicted_output.end(),
                 results_predicted_output.begin());
            copy(this->models[results_likeliest].results_output_variance.begin(),
                 this->models[results_likeliest].results_output_variance.end(),
                 results_output_variance.begin());
        } else {
            results_predicted_output.assign(dimension_output, 0.0);
            results_output_variance.assign(dimension_output, 0.0);
            
            int i(0);
            for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
                for (int d=0; d<dimension_output; d++) {
                    results_predicted_output[d] += results_normalized_likelihoods[i] * it->second.results_predicted_output[d];
                    results_output_variance[d] += results_normalized_likelihoods[i] * it->second.results_output_variance[d];
                }
                i++;
            }
        }
    }
}

#pragma mark -
#pragma mark File IO
JSONNode HierarchicalHMM::to_json() const
{
    JSONNode json_hhmm(JSON_NODE);
    json_hhmm.set_name("HierarchicalHMM");
    json_hhmm.push_back(JSONNode("bimodal", bimodal_));
    json_hhmm.push_back(JSONNode("dimension", dimension()));
    if (bimodal_)
        json_hhmm.push_back(JSONNode("dimension_input", dimension_input()));
    JSONNode json_stopcriterion(JSON_NODE);
    json_stopcriterion.set_name("EMStopCriterion");
    json_stopcriterion.push_back(JSONNode("minsteps", get_EM_minSteps()));
    json_stopcriterion.push_back(JSONNode("maxsteps", get_EM_maxSteps()));
    json_stopcriterion.push_back(JSONNode("percentchg", get_EM_percentChange()));
    json_hhmm.push_back(json_stopcriterion);
    json_hhmm.push_back(JSONNode("likelihoodwindow", get_likelihoodwindow()));
    json_hhmm.push_back(JSONNode("estimatemeans", get_estimateMeans()));
    json_hhmm.push_back(JSONNode("size", models.size()));
    json_hhmm.push_back(JSONNode("performancemode", int(performanceMode_)));
    json_hhmm.push_back(JSONNode("nbstates", get_nbStates()));
    json_hhmm.push_back(JSONNode("nbmixturecomponents", get_nbMixtureComponents()));
    json_hhmm.push_back(JSONNode("varianceoffset_relative", get_varianceOffset_relative()));
    json_hhmm.push_back(JSONNode("varianceoffset_absolute", get_varianceOffset_absolute()));
    json_hhmm.push_back(JSONNode("covariance_mode", get_covariance_mode()));
    json_hhmm.push_back(JSONNode("regression_estimator", get_regression_estimator()));
    
    // Add Models
    JSONNode json_models(JSON_ARRAY);
    for (const_model_iterator it = models.begin(); it != models.end(); ++it)
    {
        JSONNode json_model(JSON_NODE);
        json_model.push_back(it->first.to_json());
        json_model.push_back(it->second.to_json());
        json_models.push_back(json_model);
    }
    json_models.set_name("models");
    json_hhmm.push_back(json_models);
    
    // High Level transition parameters
    json_hhmm.push_back(JSONNode("incrementallearning", incrementalLearning_));
    // Prior
    JSONNode json_prior(JSON_ARRAY);
    json_prior.set_name("prior");
    for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it)
        json_prior.push_back(JSONNode("", prior.at(it->first)));
    json_hhmm.push_back(json_prior);
    
    // Exit Probabilities
    JSONNode json_exit(JSON_ARRAY);
    json_exit.set_name("exit");
    for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it)
        json_exit.push_back(JSONNode("", exitTransition.at(it->first)));
    json_hhmm.push_back(json_exit);
    
    // Transition Matrix
    JSONNode json_transition(JSON_ARRAY);
    json_transition.set_name("transition");
    for (const_model_iterator it1 = this->models.begin() ; it1 != this->models.end() ; ++it1)
        for (const_model_iterator it2 = this->models.begin() ; it2 != this->models.end() ; ++it2)
            json_transition.push_back(JSONNode("", transition.at(it1->first).at(it2->first)));
    json_hhmm.push_back(json_transition);
    
    return json_hhmm;
}


void HierarchicalHMM::from_json(JSONNode root)
{
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Number of modalities
        root_it = root.find("bimodal");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type for node 'bimodal': was expecting 'JSON_BOOL'", root_it->name());
        if (bimodal_ != root_it->as_bool()) {
            if (bimodal_)
                throw JSONException("Trying to read an unimodal model in a bimodal model.", root.name());
            else
                throw JSONException("Trying to read a bimodal model in an unimodal model.", root.name());
        }
        
        // Get Dimension
        root_it = root.find("dimension");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'dimension': was expecting 'JSON_NUMBER'", root_it->name());
        this->referenceModel_.dimension_ = static_cast<unsigned int>(root_it->as_int());
        
        // Get Input Dimension if bimodal
        if (bimodal_){
            root_it = root.find("dimension_input");
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'dimension_input': was expecting 'JSON_NUMBER'", root_it->name());
            this->referenceModel_.dimension_input_ = static_cast<unsigned int>(root_it->as_int());
        }
        
        // Get EM Algorithm stop criterion
        root_it = root.find("EMStopCriterion");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NODE)
            throw JSONException("Wrong type for node 'EMStopCriterion': was expecting 'JSON_NODE'", root_it->name());
        JSONNode json_stopcriterion = *root_it;
        JSONNode::const_iterator crit_it = json_stopcriterion.begin();
        if (crit_it == root.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "minsteps")
            throw JSONException("Wrong name: was expecting 'minsteps'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        set_EM_minSteps(static_cast<unsigned int>(crit_it->as_int()));
        crit_it++;
        
        if (crit_it == root.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "maxsteps")
            throw JSONException("Wrong name: was expecting 'maxsteps'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        set_EM_maxSteps(static_cast<unsigned int>(crit_it->as_int()));
        crit_it++;
        
        if (crit_it == root.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "percentchg")
            throw JSONException("Wrong name: was expecting 'percentchg'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        set_EM_percentChange(crit_it->as_float());
        
        // Get likelihood window size
        root_it = root.find("likelihoodwindow");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'likelihoodwindow': was expecting 'JSON_NUMBER'", root_it->name());
            this->set_likelihoodwindow(static_cast<unsigned int>(root_it->as_int()));
        }
        
        // Get If Estimate Means
        root_it = root.find("estimatemeans");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_BOOL)
                throw JSONException("Wrong type for node 'estimatemeans': was expecting 'JSON_BOOL'", root_it->name());
            this->set_estimateMeans(root_it->as_bool());
        }
        
        // Get Size: Number of Models
        root_it = root.find("size");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'size': was expecting 'JSON_NUMBER'", root_it->name());
        unsigned int numModels = static_cast<unsigned int>(root_it->as_int());
        
        // Get Play Mode
        root_it = root.find("performancemode");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'performancemode': was expecting 'JSON_NUMBER'", root_it->name());
        performanceMode_ = (root_it->as_int() > 0) ? MIXTURE : LIKELIEST;
        
        // Get Number of States
        root_it = root.find("nbstates");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'nbstates': was expecting 'JSON_NUMBER'", root_it->name());
        set_nbStates(static_cast<int>(root_it->as_int()));
        
        // Get Number of Mixture Components
        root_it = root.find("nbmixturecomponents");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'nbmixturecomponents': was expecting 'JSON_NUMBER'", root_it->name());
        set_nbMixtureComponents(static_cast<int>(root_it->as_int()));
        
        // Get Covariance Offset (Relative to data variance)
        root_it = root.find("varianceoffset_relative");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'varianceoffset_relative': was expecting 'JSON_NUMBER'", root_it->name());
        double relvar = root_it->as_float();
        
        // Get Covariance Offset (Minimum value)
        root_it = root.find("varianceoffset_absolute");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'varianceoffset_absolute': was expecting 'JSON_NUMBER'", root_it->name());
        set_varianceOffset(relvar, root_it->as_float());
        
        // Get Covariance mode
        root_it = root.find("covariance_mode");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'covariance_mode': was expecting 'JSON_NUMBER'", root_it->name());
            set_covariance_mode(static_cast<GaussianDistribution::COVARIANCE_MODE>(root_it->as_int()));
        } else {
            set_covariance_mode(GaussianDistribution::FULL);
        }
        
        // Get Regression Estimator
        root_it = root.find("regression_estimator");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'regression_estimator': was expecting 'JSON_NUMBER'", root_it->name());
        set_regression_estimator(REGRESSION_ESTIMATOR(root_it->as_int()));
        
        // Get Models
        models.clear();
        root_it = root.find("models");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'models': was expecting 'JSON_ARRAY'", root_it->name());
        for (unsigned int i=0 ; i<numModels ; i++)
        {
            // Get Label
            JSONNode::const_iterator array_it = (*root_it)[i].begin();
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete", array_it->name());
            if (array_it->name() != "label")
                throw JSONException("Wrong name: was expecting 'label'", array_it->name());
            if (array_it->type() != JSON_NODE)
                throw JSONException("Wrong type: was expecting 'JSON_NODE'", array_it->name());
            Label l;
            l.from_json(*array_it);
            ++array_it;
            
            // Get Phrase Content
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete", array_it->name());
            if (array_it->name() != "HMM")
                throw JSONException("Wrong name: was expecting 'HMM'", array_it->name());
            if (array_it->type() != JSON_NODE)
                throw JSONException("Wrong type: was expecting 'JSON_NODE'", array_it->name());
            models[l] = this->referenceModel_;
            models[l].from_json(*array_it);
        }
        if (numModels != models.size())
            throw JSONException("Number of models does not match", root.name());
        
        // Get Learning Mode
        root_it = root.find("incrementallearning");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_BOOL)
                throw JSONException("Wrong type for node 'incrementallearning': was expecting 'JSON_BOOL'", root_it->name());
            incrementalLearning_ = root_it->as_bool();
        }
        
        // Get High-level Prior
        root_it = root.find("prior");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'prior': was expecting 'JSON_ARRAY'", root_it->name());
        prior.clear();
        JSONNode::const_iterator array_it = (*root_it).begin();
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it) {
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            prior[it->first] = double(array_it->as_float());
            ++array_it;
        }
        
        // Get High-level Exit Probabilities
        root_it = root.find("exit");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'exit': was expecting 'JSON_ARRAY'", root_it->name());
        exitTransition.clear();
        array_it = (*root_it).begin();
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it) {
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            exitTransition[it->first] = double(array_it->as_float());
            ++array_it;
        }
        
        // Get High-level Transition Matrix
        root_it = root.find("transition");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'transition': was expecting 'JSON_ARRAY'", root_it->name());
        transition.clear();
        array_it = (*root_it).begin();
        for (const_model_iterator it1 = this->models.begin() ; it1 != this->models.end() ; ++it1) {
            for (const_model_iterator it2 = this->models.begin() ; it2 != this->models.end() ; ++it2)
            {
                if (root_it == root.end())
                    throw JSONException("JSON Node is incomplete", root_it->name());
                transition[it1->first][it2->first] = double(array_it->as_float());
                ++array_it;
            }
        }
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}

#pragma mark > Conversion & Extraction
void HierarchicalHMM::make_bimodal(unsigned int dimension_input)
{
    if (is_training())
        throw runtime_error("Cannot convert model during Training");
    if (bimodal_)
        throw runtime_error("The model is already bimodal");
    if (dimension_input >= dimension())
        throw out_of_range("Request input dimension exceeds the current dimension");
    
    try {
        this->referenceModel_.make_bimodal(dimension_input);
    } catch (exception const& e) {
    }
    bimodal_ = true;
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.make_bimodal(dimension_input);
    }
    set_trainingSet(NULL);
    results_predicted_output.resize(dimension() - this->dimension_input());
    results_output_variance.resize(dimension() - this->dimension_input());
}

void HierarchicalHMM::make_unimodal()
{
    if (is_training())
        throw runtime_error("Cannot convert model during Training");
    if (!bimodal_)
        throw runtime_error("The model is already unimodal");
    this->referenceModel_.make_unimodal();
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.make_unimodal();
    }
    set_trainingSet(NULL);
    results_predicted_output.clear();
    results_output_variance.clear();
    bimodal_ = false;
}

HierarchicalHMM HierarchicalHMM::extract_submodel(vector<unsigned int>& columns) const
{
    if (is_training())
        throw runtime_error("Cannot extract model during Training");
    if (columns.size() > this->dimension())
        throw out_of_range("requested number of columns exceeds the dimension of the current model");
    for (unsigned int column=0; column<columns.size(); ++column) {
        if (columns[column] >= this->dimension())
            throw out_of_range("Some column indices exceeds the dimension of the current model");
    }
    HierarchicalHMM target_model(*this);
    target_model.set_trainingSet(NULL);
    target_model.set_trainingCallback(monitor_training, (void*)this);
    target_model.bimodal_ = false;
    target_model.referenceModel_ = this->referenceModel_.extract_submodel(columns);
    for (model_iterator it=target_model.models.begin(); it != target_model.models.end(); ++it) {
        it->second = this->models.at(it->first).extract_submodel(columns);
    }
    return target_model;
}

HierarchicalHMM HierarchicalHMM::extract_submodel_input() const
{
    if (!bimodal_)
        throw runtime_error("The model needs to be bimodal");
    vector<unsigned int> columns_input(dimension_input());
    for (unsigned int i=0; i<dimension_input(); ++i) {
        columns_input[i] = i;
    }
    return extract_submodel(columns_input);
}

HierarchicalHMM HierarchicalHMM::extract_submodel_output() const
{
    if (!bimodal_)
        throw runtime_error("The model needs to be bimodal");
    vector<unsigned int> columns_output(dimension() - dimension_input());
    for (unsigned int i=dimension_input(); i<dimension(); ++i) {
        columns_output[i-dimension_input()] = i;
    }
    return extract_submodel(columns_output);
}

HierarchicalHMM HierarchicalHMM::extract_inverse_model() const
{
    if (!bimodal_)
        throw runtime_error("The model needs to be bimodal");
    vector<unsigned int> columns(dimension());
    for (unsigned int i=0; i<dimension()-dimension_input(); ++i) {
        columns[i] = i+dimension_input();
    }
    for (unsigned int i=dimension()-dimension_input(), j=0; i<dimension(); ++i, ++j) {
        columns[i] = j;
    }
    HierarchicalHMM target_model = extract_submodel(columns);
    target_model.make_bimodal(dimension()-dimension_input());
    return target_model;
}
