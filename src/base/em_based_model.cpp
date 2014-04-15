//
// em_based_learning_model.cpp
//
// Machine learning model based on the EM algorithm
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#include <cmath>
#include "em_based_model.h"

#pragma mark -
#pragma mark Constructors
EMBasedModel::EMBasedModel(rtml_flags flags,
                                           TrainingSet *trainingSet)
: BaseModel(flags, trainingSet)
{
    results.instant_likelihood = 0.0;
    results.logLikelihood = 0.0;
    stopcriterion_.minSteps = EM_MODEL_DEFAULT_EMSTOP_MINSTEPS;
    stopcriterion_.maxSteps = EM_MODEL_DEFAULT_EMSTOP_MAXSTEPS;
    stopcriterion_.percentChg = EM_MODEL_DEFAULT_EMSTOP_PERCENT_CHG;
    likelihoodBuffer_.resize(EM_MODEL_DEFAULT_LIKELIHOOD_WINDOW);
}

EMBasedModel::EMBasedModel(EMBasedModel const& src) : BaseModel(src)
{
    this->_copy(this, src);
}

EMBasedModel& EMBasedModel::operator=(EMBasedModel const& src)
{
    if(this != &src)
    {
        _copy(this, src);
    }
    return *this;
};

void EMBasedModel::_copy(EMBasedModel *dst,
                   EMBasedModel const& src)
{
    BaseModel::_copy(dst, src);
    dst->stopcriterion_.minSteps = src.stopcriterion_.minSteps;
    dst->stopcriterion_.maxSteps = src.stopcriterion_.maxSteps;
    dst->stopcriterion_.percentChg = src.stopcriterion_.percentChg;
    dst->likelihoodBuffer_.resize(src.likelihoodBuffer_.size());
    dst->likelihoodBuffer_.clear();
}

EMBasedModel::~EMBasedModel()
{}

#pragma mark -
#pragma mark Training
int EMBasedModel::train()
{
    if (!this->trainingSet)
        throw runtime_error("No training Set is Connected");
    
    if (this->trainingSet->is_empty())
        throw runtime_error("No training data");
    
#if __cplusplus > 199711L
    this->trainingMutex.lock();
#endif
    
    this->initTraining();
    
    double log_prob(log(0.)), old_log_prob;
    int nbIterations(0);
    
    do {
        old_log_prob = log_prob;
        log_prob = this->train_EM_update();
        ++nbIterations;
        
        if (stopcriterion_.maxSteps > stopcriterion_.minSteps)
            this->trainingProgression = float(nbIterations) / float(stopcriterion_.maxSteps);
        else
            this->trainingProgression = float(nbIterations) / float(stopcriterion_.minSteps);
        
        if (isnan(100.*fabs((log_prob-old_log_prob)/old_log_prob)) && (nbIterations > 1)) {
#if __cplusplus > 199711L
            this->trainingMutex.unlock();
#endif
            // TODO: Integrate exception pointer???
            if (this->trainingCallback_) {
                this->trainingCallback_(this, TRAINING_ERROR, this->trainingExtradata_);
                return -1;
            }
            else
                throw runtime_error("Training Error: No convergence! Try again... (maybe change nb of states or increase covarianceOffset)");
        }
    } while (!train_EM_stop(nbIterations, log_prob, old_log_prob));
    
    this->finishTraining();
    this->trained = true;
    this->trainingSet->set_unchanged();
    
#if __cplusplus > 199711L
    this->trainingMutex.unlock();
#endif
    return nbIterations;
}

#pragma mark -
#pragma mark EM Stop Criterion
int EMBasedModel::get_EM_minSteps() const
{
    return stopcriterion_.minSteps;
}

int EMBasedModel::get_EM_maxSteps() const
{
    return stopcriterion_.maxSteps;
}

double EMBasedModel::get_EM_percentChange() const
{
    return stopcriterion_.percentChg;
}

void EMBasedModel::set_EM_minSteps(int steps)
{
    if (steps < 1) throw invalid_argument("Minimum number of EM steps must be > 0");
    
    stopcriterion_.minSteps = steps;
}

void EMBasedModel::set_EM_maxSteps(int steps)
{
    if (steps < 0) throw invalid_argument("Maximum number of EM steps must be >= 0");
    
    stopcriterion_.maxSteps = steps;
}

void EMBasedModel::set_EM_percentChange(double logLikelihoodPercentChg)
{
    if (logLikelihoodPercentChg > 0) {
        stopcriterion_.percentChg = logLikelihoodPercentChg;
    } else {
        throw invalid_argument("Max loglikelihood difference for EM stop criterion must be > 0");
    }
}

bool EMBasedModel::train_EM_stop(int step, double log_prob, double old_log_prob) const
{
    if (stopcriterion_.maxSteps > stopcriterion_.minSteps)
        return (step >= stopcriterion_.maxSteps);
    else
        return (step >= stopcriterion_.minSteps) && (100.*fabs((log_prob - old_log_prob) / log_prob) < stopcriterion_.percentChg);
}

#pragma mark -
#pragma mark Likelihood Buffer
unsigned int EMBasedModel::get_likelihoodBufferSize() const
{
    return likelihoodBuffer_.size();
}

void EMBasedModel::set_likelihoodBufferSize(unsigned int likelihoodBufferSize)
{
    if (likelihoodBufferSize < 1) throw invalid_argument("Likelihood Buffer size must be > 1");
    likelihoodBuffer_.resize(likelihoodBufferSize);
}

void EMBasedModel::updateLikelihoodBuffer(double instantLikelihood)
{
    likelihoodBuffer_.push(log(instantLikelihood));
    results.instant_likelihood = instantLikelihood;
    results.logLikelihood = 0.0;
    unsigned int bufSize = likelihoodBuffer_.size_t();
    for (unsigned int i=0; i<bufSize; i++) {
        results.logLikelihood += likelihoodBuffer_(0, i);
    }
    results.logLikelihood /= double(bufSize);
}

void EMBasedModel::initPlaying()
{
    BaseModel::initPlaying();
    likelihoodBuffer_.clear();
}

#pragma mark -
#pragma mark File IO
JSONNode EMBasedModel::to_json() const
{
    JSONNode json_model(JSON_NODE);
    json_model.set_name("EMBasedModel");
    
    // Write Parent: Learning Model
    JSONNode json_learningmodel = BaseModel::to_json();
    json_learningmodel.set_name("LearningModel");
    json_model.push_back(json_learningmodel);
    
    JSONNode json_stopcriterion(JSON_NODE);
    json_stopcriterion.set_name("EMStopCriterion");
    json_stopcriterion.push_back(JSONNode("minsteps", stopcriterion_.minSteps));
    json_stopcriterion.push_back(JSONNode("maxsteps", stopcriterion_.maxSteps));
    json_stopcriterion.push_back(JSONNode("percentchg", stopcriterion_.percentChg));
    json_model.push_back(json_stopcriterion);
    json_model.push_back(JSONNode("likelihoodwindow", likelihoodBuffer_.size()));
    
    return json_model;
}

void EMBasedModel::from_json(JSONNode root)
{
    try {
        assert(root.type() == JSON_NODE);
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Parent: LearningModel
        assert(root_it != root.end());
        assert(root_it->name() == "LearningModel");
        assert(root_it->type() == JSON_NODE);
        BaseModel::from_json(*root_it);
        ++root_it;
        
        // Get EM Algorithm stop criterion
        assert(root_it != root.end());
        assert(root_it->name() == "EMStopCriterion");
        assert(root_it->type() == JSON_NODE);
        JSONNode json_stopcriterion = *root_it;
        JSONNode::const_iterator crit_it = json_stopcriterion.begin();
        assert(crit_it != json_stopcriterion.end());
        assert(crit_it->name() == "minsteps");
        assert(crit_it->type() == JSON_NUMBER);
        stopcriterion_.minSteps = crit_it->as_int();
        crit_it++;
        
        assert(crit_it != json_stopcriterion.end());
        assert(crit_it->name() == "maxsteps");
        assert(crit_it->type() == JSON_NUMBER);
        stopcriterion_.maxSteps = crit_it->as_int();
        crit_it++;
        
        assert(crit_it != json_stopcriterion.end());
        assert(crit_it->name() == "percentchg");
        assert(crit_it->type() == JSON_NUMBER);
        stopcriterion_.percentChg = crit_it->as_float();
        
        root_it++;
        
        // Get likelihood window size
        assert(root_it != root.end());
        assert(root_it->name() == "likelihoodwindow");
        assert(root_it->type() == JSON_NUMBER);
        this->set_likelihoodBufferSize((unsigned int)(root_it->as_int()));
        root_it++;
        
    } catch (JSONException &e) {
        throw JSONException(e);
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}