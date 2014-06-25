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
        dimension_ = this->trainingSet->dimension();
        if (bimodal_)
            dimension_input_ = this->trainingSet->dimension_input();
    } else {
        dimension_ = 1;
        dimension_input_ = 0;
    }
    results_instant_likelihood = 0.0;
    results_log_likelihood = 0.0;
    stopcriterion.minSteps = EM_MODEL_DEFAULT_EMSTOP_MINSTEPS;
    stopcriterion.maxSteps = EM_MODEL_DEFAULT_EMSTOP_MAXSTEPS;
    stopcriterion.percentChg = EM_MODEL_DEFAULT_EMSTOP_PERCENT_CHG;
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
    dst->stopcriterion.minSteps = src.stopcriterion.minSteps;
    dst->stopcriterion.maxSteps = src.stopcriterion.maxSteps;
    dst->stopcriterion.percentChg = src.stopcriterion.percentChg;
    dst->likelihoodBuffer_.resize(src.likelihoodBuffer_.size());
    dst->likelihoodBuffer_.clear();
}

ProbabilisticModel::~ProbabilisticModel()
{
    if (trainingSet)
        trainingSet->set_parent(NULL);
}

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
        dimension_ = this->trainingSet->dimension();
        if (bimodal_)
            dimension_input_ = this->trainingSet->dimension_input();
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
        dimension_ = trainingSet->dimension();
        this->allocate();
        return;
    }
    if (bimodal_ && attribute == "dimension_input") {
        dimension_input_ = trainingSet->dimension_input();
        this->allocate();
        return;
    }
}

unsigned int ProbabilisticModel::dimension() const
{
    return dimension_;
}

unsigned int ProbabilisticModel::dimension_input() const
{
    if (!bimodal_)
        throw runtime_error("The model is not bimodal");
    return dimension_input_;
}

bool ProbabilisticModel::train_EM_hasConverged(int step, double log_prob, double old_log_prob) const
{
    if (stopcriterion.maxSteps > stopcriterion.minSteps)
        return (step >= stopcriterion.maxSteps);
    else
        return (step >= stopcriterion.minSteps) && (100.*fabs((log_prob - old_log_prob) / log_prob) < stopcriterion.percentChg);
}

unsigned int ProbabilisticModel::get_likelihoodwindow() const
{
    return likelihoodBuffer_.size();
}

void ProbabilisticModel::set_likelihoodwindow(unsigned int likelihoodwindow)
{
    if (likelihoodwindow < 1) throw invalid_argument("Likelihood Buffer size must be > 1");
    likelihoodBuffer_.resize(likelihoodwindow);
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
        bool trainingError(false);
        old_log_prob = log_prob;
        try {
            log_prob = this->train_EM_update();
        } catch (exception &e) {
            trainingError = true;
        }
        
        if (isnan(100.*fabs((log_prob-old_log_prob)/old_log_prob)) && (nbIterations > 1))
            trainingError = true;
        
        if (trainingError) {
#if __cplusplus > 199711L
            this->trainingMutex.unlock();
#endif
            // TODO: Integrate exception pointer???
            if (this->trainingCallback_) {
                this->trainingCallback_(this, TRAINING_ERROR, this->trainingExtradata_);
            }
#if __cplusplus > 199711L
            return -1;
#else
            throw runtime_error("Training Error: No convergence! (maybe change nb of states or increase covarianceOffset)");
#endif
        }
        
        ++nbIterations;
        
        if (stopcriterion.maxSteps > stopcriterion.minSteps)
            this->trainingProgression = float(nbIterations) / float(stopcriterion.maxSteps);
        else
            this->trainingProgression = float(nbIterations) / float(stopcriterion.minSteps);
        
        if (this->trainingCallback_) {
            this->trainingCallback_(this, TRAINING_RUN, this->trainingExtradata_);
        }
    } while (!train_EM_hasConverged(nbIterations, log_prob, old_log_prob));
    
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
    json_stopcriterion.push_back(JSONNode("minsteps", stopcriterion.minSteps));
    json_stopcriterion.push_back(JSONNode("maxsteps", stopcriterion.maxSteps));
    json_stopcriterion.push_back(JSONNode("percentchg", stopcriterion.percentChg));
    json_model.push_back(json_stopcriterion);
    json_model.push_back(JSONNode("likelihoodwindow", likelihoodBuffer_.size()));
    
    return json_model;
}

void ProbabilisticModel::from_json(JSONNode root)
{
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Flags
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "flags")
            throw JSONException("Wrong name: was expecting 'flags'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        if(this->flags_ != static_cast<rtml_flags>(root_it->as_int())) {
            throw JSONException("The flags of the model to read does not match the flags the current instance.", root.name());
        }
        ++root_it;
        
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
        dimension_ = static_cast<unsigned int>(root_it->as_int());
        ++root_it;
        
        // Get Input Dimension if bimodal
        if (bimodal_){
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->name() != "dimension_input")
                throw JSONException("Wrong name: was expecting 'dimension_input'", root_it->name());
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
            dimension_input_ = static_cast<unsigned int>(root_it->as_int());
            ++root_it;
        }
        
        this->allocate();
        
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
        stopcriterion.minSteps = static_cast<unsigned int>(crit_it->as_int());
        crit_it++;
        
        if (crit_it == root.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "maxsteps")
            throw JSONException("Wrong name: was expecting 'maxsteps'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        stopcriterion.maxSteps = static_cast<unsigned int>(crit_it->as_int());
        crit_it++;
        
        if (crit_it == root.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "percentchg")
            throw JSONException("Wrong name: was expecting 'percentchg'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        stopcriterion.percentChg = static_cast<double>(crit_it->as_float());
        
        root_it++;
        
        // Get likelihood window size
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "likelihoodwindow")
            throw JSONException("Wrong name: was expecting 'likelihoodwindow'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        this->set_likelihoodwindow(static_cast<unsigned int>(root_it->as_int()));
        root_it++;
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}