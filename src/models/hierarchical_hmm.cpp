//
// hierarchical_hmm.cpp
//
// Hierarchical Hidden Markov Model
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#include "hierarchical_hmm.h"

#pragma mark -
#pragma mark Constructors
HierarchicalHMM::HierarchicalHMM(rtml_flags flags,
                                 TrainingSet *_globalTrainingSet)
: ModelGroup< HMM >(flags, _globalTrainingSet)
{
    incrementalLearning_ = HHMM_DEFAULT_INCREMENTALLEARNING;
    forwardInitialized_ = false;
}

HierarchicalHMM::~HierarchicalHMM()
{
    prior.clear();
    transition.clear();
    exitTransition.clear();
    V1_.clear();
    V2_.clear();
}

#pragma mark -
#pragma mark Get & Set
int HierarchicalHMM::get_nbStates() const
{
    return this->referenceModel_.get_nbStates();
}

void HierarchicalHMM::set_nbStates(int nbStates_)
{
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
    this->referenceModel_.set_nbMixtureComponents(nbMixtureComponents_);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_nbMixtureComponents(nbMixtureComponents_);
    }
}

float  HierarchicalHMM::get_covarianceOffset() const
{
    return this->referenceModel_.get_covarianceOffset();
}

void   HierarchicalHMM::set_covarianceOffset(float covarianceOffset_)
{
    this->referenceModel_.set_covarianceOffset(covarianceOffset_);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_covarianceOffset(covarianceOffset_);
    }
}

int HierarchicalHMM::get_EM_minSteps() const
{
    return this->referenceModel_.get_EM_minSteps();
}

int HierarchicalHMM::get_EM_maxSteps() const
{
    return this->referenceModel_.get_EM_maxSteps();
}

double HierarchicalHMM::get_EM_percentChange() const
{
    return this->referenceModel_.get_EM_percentChange();
}

void HierarchicalHMM::set_EM_minSteps(int steps)
{
    this->referenceModel_.set_EM_minSteps(steps);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_EM_minSteps(steps);
    }
}

void HierarchicalHMM::set_EM_maxSteps(int steps)
{
    this->referenceModel_.set_EM_maxSteps(steps);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_EM_maxSteps(steps);
    }
}

void HierarchicalHMM::set_EM_percentChange(double logLikPercentChg_)
{
    this->referenceModel_.set_EM_percentChange(logLikPercentChg_);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_EM_percentChange(logLikPercentChg_);
    }
}

unsigned int HierarchicalHMM::get_likelihoodBufferSize() const
{
    return this->referenceModel_.get_likelihoodBufferSize();
}

void HierarchicalHMM::set_likelihoodBufferSize(unsigned int likelihoodBufferSize_)
{
    this->referenceModel_.set_likelihoodBufferSize(likelihoodBufferSize_);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_likelihoodBufferSize(likelihoodBufferSize_);
    }
}

bool HierarchicalHMM::get_estimateMeans() const
{
    return this->referenceModel_.estimateMeans_;
}

void HierarchicalHMM::set_estimateMeans(bool _estimateMeans)
{
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
    this->referenceModel_.set_transitionMode(transMode_str);
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.set_transitionMode(transMode_str);
    }
}

void HierarchicalHMM::addExitPoint(int state, float proba)
{
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
    if (learningMode == "incremental") {
        incrementalLearning_ = true;
    } else if (learningMode == "ergodic") {
        incrementalLearning_ = false;
    } else {
        throw invalid_argument("'learningMode' should be 'incremental' or 'ergodic'");
    }
}


double* HierarchicalHMM::get_prior() const
{
    double *prior_ = new double[this->size()];
    int l(0);
    for (const_model_iterator it = this->models.begin(); it != this->models.end(); ++it) {
        prior_[l++] = this->prior.at(it->first);
    }
    return prior_;
}


void HierarchicalHMM::set_prior(double *prior){
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


double* HierarchicalHMM::get_transition() const
{
    unsigned int nbPrimitives = this->size();
    double *trans_ = new double[nbPrimitives*nbPrimitives];
    int l(0);
    
    for (const_model_iterator srcit = this->models.begin(); srcit != this->models.end(); ++srcit) {
        for (const_model_iterator dstit = this->models.begin(); dstit != this->models.end(); ++dstit) {
            trans_[l++] = this->transition.at(srcit->first).at(dstit->first);
        }
    }
    return trans_;
}


void HierarchicalHMM::set_transition(double *trans) {
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


double* HierarchicalHMM::get_exitTransition() const
{
    double *exittrans_ = new double[this->size()];
    int l(0);
    
    for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it)
    {
        exittrans_[l++] = this->exitTransition.at(it->first);
    }
    return exittrans_;
}


void HierarchicalHMM::set_exitTransition(double *exittrans)
{
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
        for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; ++dstit)
            transition[srcit->first][dstit->first] /= sumTrans;
    }
    for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; ++srcit)
        prior[srcit->first] /= sumPrior;
}


void HierarchicalHMM::setOneTransition(Label srcSegmentLabel, Label dstSegmentLabel, double proba)
{
    transition[srcSegmentLabel][dstSegmentLabel] = min(proba, 1.);
    normalizeTransitions();
    // TODO: absolute/relative mode?
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
    int oldNbPrim = prior.size();
    int nbPrimitives = this->size();
    
    if (oldNbPrim>0)
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
    int oldNbPrim = prior.size();
    int nbPrimitives = this->size();
    
    if (oldNbPrim>0)
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
HierarchicalHMM::model_iterator HierarchicalHMM::forward_init(const float* observation, double* modelLikelihoods)
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
            it->second.alpha_h[0][0] *= it->second.obsProb_input(observation, 0);
        } else {
            it->second.alpha_h[0][0] *= it->second.obsProb(observation, 0);
        }
        it->second.results.instant_likelihood = it->second.alpha_h[0][0] ;
        norm_const += it->second.alpha_h[0][0] ;
    }
    
    // Normalize Alpha variables
    for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
        int N = it->second.get_nbStates();
        for (int e=0 ; e<3 ; e++)
            for (int k=0 ; k<N ; k++)
                it->second.alpha_h[e][k] /= norm_const;
    }
    
    model_iterator likeliestModel(this->models.begin());
    double maxLikelihood(0.0);
    int l(0);
    
    norm_const = 0.0;
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        if (it->second.results.instant_likelihood > maxLikelihood) {
            likeliestModel = it;
            maxLikelihood = it->second.results.instant_likelihood;
        }
        
        it->second.updateLikelihoodBuffer(it->second.results.instant_likelihood);
        norm_const += it->second.results.instant_likelihood;
        modelLikelihoods[l++] = it->second.results.instant_likelihood;
    }
    
    l = 0;
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.results_hmm.likelihoodnorm = it->second.results.instant_likelihood / norm_const;
        modelLikelihoods[l++] = it->second.results_hmm.likelihoodnorm;
    }
    
    forwardInitialized_ = true;
    return likeliestModel;
}

HierarchicalHMM::model_iterator HierarchicalHMM::forward_update(const float* observation, double* modelLikelihoods)
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
        
        dstit->second.results_hmm.exitLikelihood = 0.0;
        dstit->second.results.instant_likelihood = 0.0;
        
        // end of the primitive: handle exit states
        for (int k=0 ; k<N ; ++k)
        {
            if (bimodal_)
                tmp = dstit->second.obsProb_input(observation, k) * front[k];
            else
                tmp = dstit->second.obsProb(observation, k) * front[k];
            
            dstit->second.alpha_h[2][k] = this->exitTransition[dstit->first] * dstit->second.exitProbabilities_[k] * tmp ;
            dstit->second.alpha_h[1][k] = (1 - this->exitTransition[dstit->first]) * dstit->second.exitProbabilities_[k] * tmp ;
            dstit->second.alpha_h[0][k] = (1 - dstit->second.exitProbabilities_[k]) * tmp;
            
            dstit->second.results_hmm.exitLikelihood += dstit->second.alpha_h[1][k];
            dstit->second.results.instant_likelihood += dstit->second.alpha_h[0][k] + dstit->second.alpha_h[2][k] + dstit->second.results_hmm.exitLikelihood;
            
            norm_const += tmp;
        }
        
        // TODO: update cumulative + likelihood in circularBuffer
    }
    
    // Normalize Alpha variables
    for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
        int N = it->second.get_nbStates();
        for (int e=0 ; e<3 ; e++)
            for (int k=0 ; k<N ; k++)
                it->second.alpha_h[e][k] /= norm_const;
    }
    
    model_iterator likeliestModel(this->models.begin());
    double maxLikelihood(0.0);
    int l(0.0);
    
    norm_const = 0.0;
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        if (it->second.results.instant_likelihood > maxLikelihood) {
            likeliestModel = it;
            maxLikelihood = it->second.results.instant_likelihood;
        }
        
        it->second.updateLikelihoodBuffer(it->second.results.instant_likelihood);
        norm_const += it->second.results.instant_likelihood;
        modelLikelihoods[l++] = it->second.results.instant_likelihood;
    }
    
    l = 0;
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.results_hmm.likelihoodnorm = it->second.results.instant_likelihood / norm_const;
        modelLikelihoods[l++] = it->second.results_hmm.likelihoodnorm;
    }
    
    return likeliestModel;
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
void HierarchicalHMM::initPlaying()
{
    ModelGroup<HMM>::initPlaying();
    V1_.resize(this->size()) ;
    V2_.resize(this->size()) ;
    forwardInitialized_ = false;
}

void HierarchicalHMM::play(float *observation, double *modelLikelihoods)
{
    model_iterator likeliestModel;
    if (forwardInitialized_) {
        likeliestModel = this->forward_update(observation, modelLikelihoods);
    } else {
        likeliestModel = this->forward_init(observation, modelLikelihoods);
    }
    
    if (bimodal_) {
        unsigned int dimension = this->referenceModel_.get_dimension();
        unsigned int dimension_input = this->referenceModel_.get_dimension_input();
        unsigned int dimension_output = dimension - dimension_input;
        
        for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
            it->second.regression(observation, it->second.results.predicted_output);
        }
        
        if (this->playMode_ == this->LIKELIEST) {
            copy(likeliestModel->second.results.predicted_output.begin(),
                 likeliestModel->second.results.predicted_output.end(),
                 observation + dimension_input);
        } else {
            for (int d=0; d<dimension_output; d++) {
                observation[dimension_input + d] = 0.0;
            }
            
            int i(0);
            for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
                for (int d=0; d<dimension_output; d++) {
                    observation[dimension_input+d] += modelLikelihoods[i] * it->second.results.predicted_output[d];
                }
                i++;
            }
        }
    }
    
    // Compute time progression
    // TODO: Put this in forward algorithm
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.updateTimeProgression();
    }
}

HMM::Results HierarchicalHMM::getResults(Label const& label) const
{
    if (this->models.find(label) == this->models.end())
        throw out_of_range("Class Label Does not exist");
    return this->models.at(label).results_hmm;
}

#pragma mark -
#pragma mark File IO
JSONNode HierarchicalHMM::to_json() const
{
    JSONNode json_hhmm(JSON_NODE);
    json_hhmm.set_name("HierarchicalHMM");
    json_hhmm.push_back(JSONNode("bimodal", bimodal_));
    json_hhmm.push_back(JSONNode("dimension", get_dimension()));
    if (bimodal_)
        json_hhmm.push_back(JSONNode("dimension_input", get_dimension_input()));
    JSONNode json_stopcriterion(JSON_NODE);
    json_stopcriterion.set_name("EMStopCriterion");
    json_stopcriterion.push_back(JSONNode("minsteps", get_EM_minSteps()));
    json_stopcriterion.push_back(JSONNode("maxsteps", get_EM_maxSteps()));
    json_stopcriterion.push_back(JSONNode("percentchg", get_EM_percentChange()));
    json_hhmm.push_back(json_stopcriterion);
    json_hhmm.push_back(JSONNode("likelihoodwindow", get_likelihoodBufferSize()));
    json_hhmm.push_back(JSONNode("estimateMeans", get_estimateMeans()));
    json_hhmm.push_back(JSONNode("size", models.size()));
    json_hhmm.push_back(JSONNode("playmode", int(playMode_)));
    json_hhmm.push_back(JSONNode("nbStates", get_nbStates()));
    json_hhmm.push_back(JSONNode("nbMixtureComponents", get_nbMixtureComponents()));
    json_hhmm.push_back(JSONNode("covarianceOffset", get_covarianceOffset()));
    
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
    json_hhmm.push_back(JSONNode("incrementalLearning", incrementalLearning_));
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
        assert(root.type() == JSON_NODE);
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Number of modalities
        assert(root_it != root.end());
        assert(root_it->name() == "bimodal");
        assert(root_it->type() == JSON_BOOL);
        if(bimodal_ != root_it->as_bool()) {
            if (bimodal_)
                throw JSONException("Trying to read an unimodal model in a bimodal model.", root.name());
            else
                throw JSONException("Trying to read a bimodal model in an unimodal model.", root.name());
        }
        ++root_it;
        
        // Get Dimension
        assert(root_it != root.end());
        assert(root_it->name() == "dimension");
        assert(root_it->type() == JSON_NUMBER);
        this->referenceModel_.dimension_ = root_it->as_int();
        ++root_it;
        
        // Get Input Dimension if bimodal
        if (bimodal_){
            assert(root_it != root.end());
            assert(root_it->name() == "dimension_input");
            assert(root_it->type() == JSON_NUMBER);
            this->referenceModel_.dimension_input_ = root_it->as_int();
            ++root_it;
        }

        // Get EM Algorithm stop criterion
        assert(root_it != root.end());
        assert(root_it->name() == "EMStopCriterion");
        assert(root_it->type() == JSON_NODE);
        JSONNode json_stopcriterion = *root_it;
        JSONNode::const_iterator crit_it = json_stopcriterion.begin();
        assert(crit_it != json_stopcriterion.end());
        assert(crit_it->name() == "minsteps");
        assert(crit_it->type() == JSON_NUMBER);
        set_EM_minSteps(crit_it->as_int());
        crit_it++;
        
        assert(crit_it != json_stopcriterion.end());
        assert(crit_it->name() == "maxsteps");
        assert(crit_it->type() == JSON_NUMBER);
        set_EM_maxSteps(crit_it->as_int());
        crit_it++;
        
        assert(crit_it != json_stopcriterion.end());
        assert(crit_it->name() == "percentchg");
        assert(crit_it->type() == JSON_NUMBER);
        set_EM_percentChange(crit_it->as_float());
        
        root_it++;
        
        // Get likelihood window size
        assert(root_it != root.end());
        assert(root_it->name() == "likelihoodwindow");
        assert(root_it->type() == JSON_NUMBER);
        this->set_likelihoodBufferSize((unsigned int)(root_it->as_int()));
        root_it++;

        // Get likelihood window size
        assert(root_it != root.end());
        assert(root_it->name() == "estimateMeans");
        assert(root_it->type() == JSON_BOOL);
        this->set_estimateMeans(root_it->as_bool());
        root_it++;
        
        // Get Size: Number of Models
        assert(root_it != root.end());
        assert(root_it->name() == "size");
        assert(root_it->type() == JSON_NUMBER);
        int numModels = root_it->as_int();
        ++root_it;
        
        // Get Play Mode
        assert(root_it != root.end());
        assert(root_it->name() == "playmode");
        assert(root_it->type() == JSON_NUMBER);
        playMode_ = (root_it->as_int() > 0) ? MIXTURE : LIKELIEST;
        ++root_it;
        
        // Get Number of States
        assert(root_it != root.end());
        assert(root_it->name() == "nbStates");
        assert(root_it->type() == JSON_NUMBER);
        set_nbStates(root_it->as_int());
        ++root_it;

        // Get Number of Mixture Components
        assert(root_it != root.end());
        assert(root_it->name() == "nbMixtureComponents");
        assert(root_it->type() == JSON_NUMBER);
        set_nbMixtureComponents(root_it->as_int());
        ++root_it;
        
        // Get Covariance Offset
        assert(root_it != root.end());
        assert(root_it->name() == "covarianceOffset");
        assert(root_it->type() == JSON_NUMBER);
        set_covarianceOffset(root_it->as_float());
        ++root_it;
        
        // Get Models
        models.clear();
        assert(root_it != root.end());
        assert(root_it->name() == "models");
        assert(root_it->type() == JSON_ARRAY);
        for (int i=0 ; i<numModels ; i++)
        {
            // Get Label
            JSONNode::const_iterator array_it = (*root_it)[i].begin();
            assert(array_it != root_it->end());
            assert(array_it->name() == "label");
            assert(array_it->type() == JSON_NODE);
            Label l;
            l.from_json(*array_it);
            ++array_it;
            
            // Get Phrase Content
            assert(array_it != root_it->end());
            assert(array_it->type() == JSON_NODE);
            models[l] = this->referenceModel_;
            models[l].trainingSet = NULL;
            models[l].from_json(*array_it);
        }
        assert(numModels == models.size());
        ++root_it;
        
        // Get Learning Mode
        assert(root_it != root.end());
        assert(root_it->name() == "incrementalLearning");
        assert(root_it->type() == JSON_BOOL);
        incrementalLearning_ = root_it->as_bool();
        ++root_it;
        
        // Get High-level Prior
        assert(root_it != root.end());
        assert(root_it->name() == "prior");
        assert(root_it->type() == JSON_ARRAY);
        prior.clear();
        JSONNode::const_iterator array_it = (*root_it).begin();
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it) {
            assert(array_it != root.end());
            prior[it->first] = double(array_it->as_float());
            ++array_it;
        }
        ++root_it;
        
        // Get High-level Exit Probabilities
        assert(root_it != root.end());
        assert(root_it->name() == "exit");
        assert(root_it->type() == JSON_ARRAY);
        exitTransition.clear();
        array_it = (*root_it).begin();
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it) {
            assert(array_it != root.end());
            exitTransition[it->first] = double(array_it->as_float());
            ++array_it;
        }
        ++root_it;
        
        // Get High-level Transition Matrix
        assert(root_it != root.end());
        assert(root_it->name() == "transition");
        assert(root_it->type() == JSON_ARRAY);
        transition.clear();
        array_it = (*root_it).begin();
        for (const_model_iterator it1 = this->models.begin() ; it1 != this->models.end() ; ++it1) {
            for (const_model_iterator it2 = this->models.begin() ; it2 != this->models.end() ; ++it2)
            {
                assert(array_it != root.end());
                transition[it1->first][it2->first] = double(array_it->as_float());
                ++array_it;
            }
        }
        
    } catch (JSONException &e) {
        throw JSONException(e);
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}