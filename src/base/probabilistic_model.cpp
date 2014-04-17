//
// probabilistic_model.cpp
//
// Machine learning model based on the EM algorithm
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#include <cmath>
#include "probabilistic_model.h"

#pragma mark -
#pragma mark Constructors
ProbabilisticModel::ProbabilisticModel(rtml_flags flags,
                                           TrainingSet *trainingSet)
: trainingSet(trainingSet),
trained(false),
trainingProgression(0),
flags_(flags),
bimodal_(flags & BIMODAL),
trainingCallback_(NULL)
{
    if (this->trainingSet) {
        if (bimodal_ && !trainingSet->is_bimodal())
            throw runtime_error("This model is bimodal but the training set is not. Can't Connect.");
        if (!bimodal_ && trainingSet->is_bimodal())
            throw runtime_error("This model is not bimodal but the training set is. Can't Connect.");
        dimension_ = this->trainingSet->get_dimension();
        if (bimodal_)
            dimension_input_ = this->trainingSet->get_dimension_input();
    } else {
        dimension_ = 1;
        dimension_input_ = 0;
    }
    results_instant_likelihood = 0.0;
    results_log_likelihood = 0.0;
    stopcriterion_.minSteps = EM_MODEL_DEFAULT_EMSTOP_MINSTEPS;
    stopcriterion_.maxSteps = EM_MODEL_DEFAULT_EMSTOP_MAXSTEPS;
    stopcriterion_.percentChg = EM_MODEL_DEFAULT_EMSTOP_PERCENT_CHG;
    likelihoodBuffer_.resize(EM_MODEL_DEFAULT_LIKELIHOOD_WINDOW);
}

ProbabilisticModel::ProbabilisticModel(ProbabilisticModel const& src)
{
    this->_copy(this, src);
}

ProbabilisticModel& ProbabilisticModel::operator=(ProbabilisticModel const& src)
{
    if(this != &src)
    {
        cout << "ProbabilisticModel::operator=" << endl;
        _copy(this, src);
    }
    return *this;
};

void ProbabilisticModel::_copy(ProbabilisticModel *dst,
                   ProbabilisticModel const& src)
{
    dst->flags_ = src.flags_;
    dst->trained = src.trained;
    dst->trainingSet = src.trainingSet;
    dst->trainingCallback_ = src.trainingCallback_;
    dst->trainingExtradata_ = src.trainingExtradata_;
    dst->bimodal_ = src.bimodal_;
    if (dst->bimodal_)
        dst->dimension_input_ = src.dimension_input_;
    dst->dimension_ = src.dimension_;
    dst->stopcriterion_.minSteps = src.stopcriterion_.minSteps;
    dst->stopcriterion_.maxSteps = src.stopcriterion_.maxSteps;
    dst->stopcriterion_.percentChg = src.stopcriterion_.percentChg;
    dst->likelihoodBuffer_.resize(src.likelihoodBuffer_.size());
    dst->likelihoodBuffer_.clear();
}

ProbabilisticModel::~ProbabilisticModel()
{}

#pragma mark -
#pragma mark Accessors
void ProbabilisticModel::set_trainingSet(TrainingSet *trainingSet)
{
    this->trainingSet = trainingSet;
    if (this->trainingSet) {
        if (bimodal_ && !trainingSet->is_bimodal())
            throw runtime_error("This model is bimodal but the training set is not. Can't Connect.");
        if (!bimodal_ && trainingSet->is_bimodal())
            throw runtime_error("This model is not bimodal but the training set is. Can't Connect.");
        dimension_ = this->trainingSet->get_dimension();
        if (bimodal_)
            dimension_input_ = this->trainingSet->get_dimension_input();
    } else {
        dimension_ = 1;
        dimension_input_ = 0;
    }
    this->allocate();
}

void ProbabilisticModel::notify(string attribute)
{
    if (!trainingSet) return;
    if (attribute == "dimension") {
        dimension_ = trainingSet->get_dimension();
        this->allocate();
        return;
    }
    if (bimodal_ && attribute == "dimension_input") {
        dimension_input_ = trainingSet->get_dimension_input();
        this->allocate();
        return;
    }
}

unsigned int ProbabilisticModel::get_dimension() const
{
    return dimension_;
}

unsigned int ProbabilisticModel::get_dimension_input() const
{
    if (!bimodal_)
        throw runtime_error("The model is not bimodal");
    return dimension_input_;
}

unsigned int ProbabilisticModel::get_EM_minSteps() const
{
    return stopcriterion_.minSteps;
}

unsigned int ProbabilisticModel::get_EM_maxSteps() const
{
    return stopcriterion_.maxSteps;
}

double ProbabilisticModel::get_EM_percentChange() const
{
    return stopcriterion_.percentChg;
}

void ProbabilisticModel::set_EM_minSteps(unsigned int steps)
{
    if (steps < 1) throw invalid_argument("Minimum number of EM steps must be > 0");
    
    stopcriterion_.minSteps = steps;
}

void ProbabilisticModel::set_EM_maxSteps(unsigned int steps)
{
    if (steps < 0) throw invalid_argument("Maximum number of EM steps must be >= 0");
    
    stopcriterion_.maxSteps = steps;
}

void ProbabilisticModel::set_EM_percentChange(double logLikelihoodPercentChg)
{
    if (logLikelihoodPercentChg > 0) {
        stopcriterion_.percentChg = logLikelihoodPercentChg;
    } else {
        throw invalid_argument("Max loglikelihood difference for EM stop criterion must be > 0");
    }
}

bool ProbabilisticModel::train_EM_stop(int step, double log_prob, double old_log_prob) const
{
    if (stopcriterion_.maxSteps > stopcriterion_.minSteps)
        return (step >= stopcriterion_.maxSteps);
    else
        return (step >= stopcriterion_.minSteps) && (100.*fabs((log_prob - old_log_prob) / log_prob) < stopcriterion_.percentChg);
}

unsigned int ProbabilisticModel::get_likelihoodBufferSize() const
{
    return likelihoodBuffer_.size();
}

void ProbabilisticModel::set_likelihoodBufferSize(unsigned int likelihoodBufferSize)
{
    if (likelihoodBufferSize < 1) throw invalid_argument("Likelihood Buffer size must be > 1");
    likelihoodBuffer_.resize(likelihoodBufferSize);
}

#pragma mark -
#pragma mark Training
int ProbabilisticModel::train()
{
    if (!this->trainingSet)
        throw runtime_error("No training Set is Connected");
    
    if (this->trainingSet->is_empty())
        throw runtime_error("No training data");
    
#if __cplusplus > 199711L
    this->trainingMutex.lock();
#endif
    
    this->train_EM_init();
    
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
    
    this->train_EM_terminate();
    this->trained = true;
    this->trainingSet->set_unchanged();
    
#if __cplusplus > 199711L
    this->trainingMutex.unlock();
#endif
    return nbIterations;
}

void ProbabilisticModel::train_EM_terminate()
{
    if (trainingCallback_)
        trainingCallback_(this, TRAINING_DONE, trainingExtradata_);
}

void ProbabilisticModel::set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata)
{
    trainingExtradata_ = extradata;
    trainingCallback_ = callback;
}

#pragma mark -
#pragma mark Likelihood Buffer
void ProbabilisticModel::updateLikelihoodBuffer(double instantLikelihood)
{
    likelihoodBuffer_.push(log(instantLikelihood));
    results_instant_likelihood = instantLikelihood;
    results_log_likelihood = 0.0;
    unsigned int bufSize = likelihoodBuffer_.size_t();
    for (unsigned int i=0; i<bufSize; i++) {
        results_log_likelihood += likelihoodBuffer_(0, i);
    }
    results_log_likelihood /= double(bufSize);
}

#pragma mark -
#pragma mark Performance
void ProbabilisticModel::performance_init()
{
    if (!this->trained)
        throw runtime_error("Cannot play: model has not been trained");
    likelihoodBuffer_.clear();
}

#pragma mark -
#pragma mark File IO
JSONNode ProbabilisticModel::to_json() const
{
    JSONNode json_model(JSON_NODE);
    json_model.set_name("ProbabilisticModel");
    
    json_model.push_back(JSONNode("flags", flags_));
    json_model.push_back(JSONNode("bimodal", bimodal_));
    json_model.push_back(JSONNode("dimension", dimension_));
    if (bimodal_)
        json_model.push_back(JSONNode("dimension_input", dimension_input_));
    
    JSONNode json_stopcriterion(JSON_NODE);
    json_stopcriterion.set_name("EMStopCriterion");
    json_stopcriterion.push_back(JSONNode("minsteps", stopcriterion_.minSteps));
    json_stopcriterion.push_back(JSONNode("maxsteps", stopcriterion_.maxSteps));
    json_stopcriterion.push_back(JSONNode("percentchg", stopcriterion_.percentChg));
    json_model.push_back(json_stopcriterion);
    json_model.push_back(JSONNode("likelihoodwindow", likelihoodBuffer_.size()));
    
    return json_model;
}

void ProbabilisticModel::from_json(JSONNode root)
{
    try {
        assert(root.type() == JSON_NODE);
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Flags
        assert(root_it != root.end());
        assert(root_it->name() == "flags");
        assert(root_it->type() == JSON_NUMBER);
        if(this->flags_ != static_cast<rtml_flags>(root_it->as_int())) {
            throw JSONException("The flags of the model to read does not match the flags the current instance.", root.name());
        }
        ++root_it;
        
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
        dimension_ = static_cast<unsigned int>(root_it->as_int());
        ++root_it;
        
        // Get Input Dimension if bimodal
        if (bimodal_){
            assert(root_it != root.end());
            assert(root_it->name() == "dimension_input");
            assert(root_it->type() == JSON_NUMBER);
            dimension_input_ = static_cast<unsigned int>(root_it->as_int());
            ++root_it;
        }
        
        this->allocate();
        
        // Get EM Algorithm stop criterion
        assert(root_it != root.end());
        assert(root_it->name() == "EMStopCriterion");
        assert(root_it->type() == JSON_NODE);
        JSONNode json_stopcriterion = *root_it;
        JSONNode::const_iterator crit_it = json_stopcriterion.begin();
        assert(crit_it != json_stopcriterion.end());
        assert(crit_it->name() == "minsteps");
        assert(crit_it->type() == JSON_NUMBER);
        stopcriterion_.minSteps = static_cast<unsigned int>(crit_it->as_int());
        crit_it++;
        
        assert(crit_it != json_stopcriterion.end());
        assert(crit_it->name() == "maxsteps");
        assert(crit_it->type() == JSON_NUMBER);
        stopcriterion_.maxSteps = static_cast<unsigned int>(crit_it->as_int());
        crit_it++;
        
        assert(crit_it != json_stopcriterion.end());
        assert(crit_it->name() == "percentchg");
        assert(crit_it->type() == JSON_NUMBER);
        stopcriterion_.percentChg = static_cast<double>(crit_it->as_float());
        
        root_it++;
        
        // Get likelihood window size
        assert(root_it != root.end());
        assert(root_it->name() == "likelihoodwindow");
        assert(root_it->type() == JSON_NUMBER);
        this->set_likelihoodBufferSize(static_cast<unsigned int>(root_it->as_int()));
        root_it++;
        
    } catch (JSONException &e) {
        throw JSONException(e);
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}