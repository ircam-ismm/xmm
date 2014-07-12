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
: ModelGroup< HMM >(flags|HIERARCHICAL, _globalTrainingSet)
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

double HierarchicalHMM::get_weight_regression() const
{
    return this->referenceModel_.get_weight_regression();
}

void HierarchicalHMM::set_weight_regression(double weight_regression)
{
    this->referenceModel_.set_weight_regression(weight_regression);
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.set_weight_regression(weight_regression);
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
        } else {
            results_predicted_output.assign(dimension_output, 0.0);
            
            int i(0);
            for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
                for (int d=0; d<dimension_output; d++) {
                    results_predicted_output[d] += results_normalized_likelihoods[i] * it->second.results_predicted_output[d];
                }
                i++;
            }
        }
    }
    
    // Compute time progression
    for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
        it->second.updateTimeProgression();
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
    json_hhmm.push_back(JSONNode("weight_regression", get_weight_regression()));
    
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
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "bimodal")
            throw JSONException("Wrong name: was expecting 'bimodal'", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type: was expecting 'JSON_BOOL'", root_it->name());
        if(bimodal_ != root_it->as_bool()) {
            if (bimodal_)
                throw JSONException("Trying to read an unimodal model in a bimodal model.", root.name());
            else
                throw JSONException("Trying to read a bimodal model in an unimodal model.", root.name());
        }
        ++root_it;
        
        // Get Dimension
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "dimension")
            throw JSONException("Wrong name: was expecting 'dimension'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        this->referenceModel_.dimension_ = root_it->as_int();
        ++root_it;
        
        // Get Input Dimension if bimodal
        if (bimodal_){
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->name() != "dimension_input")
                throw JSONException("Wrong name: was expecting 'dimension_input'", root_it->name());
            if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
            this->referenceModel_.dimension_input_ = root_it->as_int();
            ++root_it;
        }

        // Get EM Algorithm stop criterion
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "EMStopCriterion")
            throw JSONException("Wrong name: was expecting 'EMStopCriterion'", root_it->name());
        if (root_it->type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root_it->name());
        JSONNode json_stopcriterion = *root_it;
        JSONNode::const_iterator crit_it = json_stopcriterion.begin();
        if (crit_it == root.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "minsteps")
            throw JSONException("Wrong name: was expecting 'minsteps'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        set_EM_minSteps(crit_it->as_int());
        crit_it++;
        
        if (crit_it == root.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "maxsteps")
            throw JSONException("Wrong name: was expecting 'maxsteps'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        set_EM_maxSteps(crit_it->as_int());
        crit_it++;
        
        if (crit_it == root.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "percentchg")
            throw JSONException("Wrong name: was expecting 'percentchg'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        set_EM_percentChange(crit_it->as_float());
        
        root_it++;
        
        // Get likelihood window size
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "likelihoodwindow")
            throw JSONException("Wrong name: was expecting 'likelihoodwindow'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        this->set_likelihoodwindow((unsigned int)(root_it->as_int()));
        root_it++;

        // Get likelihood window size
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "estimatemeans")
            throw JSONException("Wrong name: was expecting 'estimatemeans'", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type: was expecting 'JSON_BOOL'", root_it->name());
        this->set_estimateMeans(root_it->as_bool());
        root_it++;
        
        // Get Size: Number of Models
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "size")
            throw JSONException("Wrong name: was expecting 'size'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        int numModels = root_it->as_int();
        ++root_it;
        
        // Get Play Mode
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "performancemode")
            throw JSONException("Wrong name: was expecting 'performancemode'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        performanceMode_ = (root_it->as_int() > 0) ? MIXTURE : LIKELIEST;
        ++root_it;
        
        // Get Number of States
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "nbstates")
            throw JSONException("Wrong name: was expecting 'nbstates'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        set_nbStates(root_it->as_int());
        ++root_it;

        // Get Number of Mixture Components
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "nbmixturecomponents")
            throw JSONException("Wrong name: was expecting 'nbmixturecomponents'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        set_nbMixtureComponents(root_it->as_int());
        ++root_it;
        
        // Get Covariance Offset
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "varianceoffset_relative")
            throw JSONException("Wrong name: was expecting 'varianceoffset_relative'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        double relvar = root_it->as_float();
        ++root_it;
        
        // Get Covariance Offset
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "varianceoffset_absolute")
            throw JSONException("Wrong name: was expecting 'varianceoffset_absolute'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        set_varianceOffset(relvar, root_it->as_float());
        ++root_it;
        
        // Get Covariance Offset
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "weight_regression")
            throw JSONException("Wrong name: was expecting 'weight_regression'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        set_weight_regression(root_it->as_float());
        ++root_it;
        
        // Get Models
        models.clear();
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "models")
            throw JSONException("Wrong name: was expecting 'models'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        for (int i=0 ; i<numModels ; i++)
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
            if (array_it->type() != JSON_NODE)
                throw JSONException("Wrong type: was expecting 'JSON_NODE'", array_it->name());
            models[l] = this->referenceModel_;
            models[l].from_json(*array_it);
        }
        if (numModels != models.size())
            throw JSONException("Number of models does not match", root.name());
        ++root_it;
        
        // Get Learning Mode
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "incrementallearning")
            throw JSONException("Wrong name: was expecting 'incrementallearning'", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type: was expecting 'JSON_BOOL'", root_it->name());
        incrementalLearning_ = root_it->as_bool();
        ++root_it;
        
        // Get High-level Prior
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "prior")
            throw JSONException("Wrong name: was expecting 'prior'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        prior.clear();
        JSONNode::const_iterator array_it = (*root_it).begin();
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it) {
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            prior[it->first] = double(array_it->as_float());
            ++array_it;
        }
        ++root_it;
        
        // Get High-level Exit Probabilities
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "exit")
            throw JSONException("Wrong name: was expecting 'exit'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        exitTransition.clear();
        array_it = (*root_it).begin();
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; ++it) {
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            exitTransition[it->first] = double(array_it->as_float());
            ++array_it;
        }
        ++root_it;
        
        // Get High-level Transition Matrix
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "transition")
            throw JSONException("Wrong name: was expecting 'transition'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
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