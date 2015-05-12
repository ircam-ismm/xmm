/*
 * probabilistic_model.cpp
 *
 * Abstract class for Probabilistic Machine learning models based on the EM algorithm
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

#include <cmath>
#include "probabilistic_model.h"

#pragma mark -
#pragma mark Constructors
xmm::ProbabilisticModel::ProbabilisticModel(xmm_flags flags,
                                       TrainingSet *trainingSet)
: trainingSet(trainingSet),
  trained(false),
  trainingProgression(0),
  flags_(flags),
  bimodal_(flags & BIMODAL),
  trainingCallbackFunction_(NULL)
#ifdef USE_PTHREAD
  ,is_training_(false),
  cancel_training_(false)
#endif
{
    if (this->trainingSet) {
        if (bimodal_ && !trainingSet->is_bimodal())
            throw std::runtime_error("This model is bimodal but the training set is not. Can't Connect.");
        if (!bimodal_ && trainingSet->is_bimodal())
            throw std::runtime_error("This model is not bimodal but the training set is. Can't Connect.");
        this->trainingSet->add_listener(this);
        dimension_ = this->trainingSet->dimension();
        if (bimodal_)
            dimension_input_ = this->trainingSet->dimension_input();
        column_names_.resize(dimension_);
        column_names_ = this->trainingSet->get_column_names();
    } else {
        dimension_ = 1;
        column_names_.resize(dimension_);
        dimension_input_ = 0;
    }
    results_instant_likelihood = 0.0;
    results_log_likelihood = 0.0;
    stopcriterion.minSteps = DEFAULT_EMSTOP_MINSTEPS;
    stopcriterion.maxSteps = DEFAULT_EMSTOP_MAXSTEPS;
    stopcriterion.percentChg = DEFAULT_EMSTOP_PERCENT_CHG();
    likelihoodBuffer_.resize(DEFAULT_LIKELIHOOD_WINDOW);
#ifdef USE_PTHREAD
    pthread_mutex_init(&trainingMutex, NULL);
#endif
}

xmm::ProbabilisticModel::ProbabilisticModel(ProbabilisticModel const& src)
{
    this->_copy(this, src);
}

xmm::ProbabilisticModel& xmm::ProbabilisticModel::operator=(ProbabilisticModel const& src)
{
    if(this != &src)
    {
        if (this->trainingSet)
            this->trainingSet->remove_listener(this);
        _copy(this, src);
    }
    return *this;
};

void xmm::ProbabilisticModel::_copy(ProbabilisticModel *dst,
                               ProbabilisticModel const& src)
{
#ifdef USE_PTHREAD
    if (src.is_training())
        throw std::runtime_error("Cannot copy model during Training");
    dst->is_training_ = false;
#endif
    dst->flags_ = src.flags_;
    dst->trained = src.trained;
    dst->trainingCallbackFunction_ = src.trainingCallbackFunction_;
    dst->trainingExtradata_ = src.trainingExtradata_;
    dst->bimodal_ = src.bimodal_;
    if (dst->bimodal_)
        dst->dimension_input_ = src.dimension_input_;
    dst->dimension_ = src.dimension_;
    dst->column_names_ = src.column_names_;
    dst->stopcriterion.minSteps = src.stopcriterion.minSteps;
    dst->stopcriterion.maxSteps = src.stopcriterion.maxSteps;
    dst->stopcriterion.percentChg = src.stopcriterion.percentChg;
    dst->likelihoodBuffer_.resize(src.likelihoodBuffer_.size());
    dst->likelihoodBuffer_.clear();
    dst->trainingSet = src.trainingSet;
    if (dst->trainingSet)
        dst->trainingSet->add_listener(dst);
}

xmm::ProbabilisticModel::~ProbabilisticModel()
{
#ifdef USE_PTHREAD
    while (this->is_training()) {}
#endif
    if (this->trainingSet)
        this->trainingSet->remove_listener(this);
}

#pragma mark -
#pragma mark Accessors
#ifdef USE_PTHREAD
bool xmm::ProbabilisticModel::is_training() const
{
    return is_training_;
}
#endif

void xmm::ProbabilisticModel::set_trainingSet(TrainingSet *trainingSet)
{
    prevent_attribute_change();
    if (this->trainingSet)
        this->trainingSet->remove_listener(this);
    this->trainingSet = trainingSet;
    if (this->trainingSet) {
        if (bimodal_ && !trainingSet->is_bimodal())
            throw std::runtime_error("This model is bimodal but the training set is not. Can't Connect.");
        if (!bimodal_ && trainingSet->is_bimodal())
            throw std::runtime_error("This model is not bimodal but the training set is. Can't Connect.");
        this->trainingSet->add_listener(this);
        dimension_ = this->trainingSet->dimension();
        if (bimodal_)
            dimension_input_ = this->trainingSet->dimension_input();
        column_names_.resize(dimension_);
        column_names_ = this->trainingSet->get_column_names();
        this->allocate();
    }
}

void xmm::ProbabilisticModel::notify(std::string attribute)
{
#ifdef USE_PTHREAD
    if (is_training())
        throw std::runtime_error("Cannot receive notifications during Training");
#endif
    if (!trainingSet) return;
    if (attribute == "dimension") {
        dimension_ = trainingSet->dimension();
        column_names_.resize(dimension_);
        this->allocate();
        return;
    }
    if (bimodal_ && attribute == "dimension_input") {
        dimension_input_ = trainingSet->dimension_input();
        this->allocate();
        return;
    }
    if (attribute == "column_names") {
        column_names_.resize(dimension_);
        column_names_ = trainingSet->get_column_names();
        return;
    }
    if (attribute == "destruction") {
        trainingSet = NULL;
        return;
    }
}

unsigned int xmm::ProbabilisticModel::dimension() const
{
    return dimension_;
}

unsigned int xmm::ProbabilisticModel::dimension_input() const
{
    if (!bimodal_)
        throw std::runtime_error("The model is not bimodal");
    return dimension_input_;
}

unsigned int xmm::ProbabilisticModel::get_likelihoodwindow() const
{
    return likelihoodBuffer_.size();
}

void xmm::ProbabilisticModel::set_likelihoodwindow(unsigned int likelihoodwindow)
{
    if (likelihoodwindow < 1) throw std::invalid_argument("Likelihood Buffer size must be > 1");
    likelihoodBuffer_.resize(likelihoodwindow);
}

std::vector<std::string> const& xmm::ProbabilisticModel::get_column_names() const
{
    return column_names_;
}

#pragma mark -
#pragma mark Training
void* xmm::ProbabilisticModel::train_func(void *context)
{
    ((ProbabilisticModel *)context)->train();
    return NULL;
}

void xmm::ProbabilisticModel::train()
{
#ifdef USE_PTHREAD
    pthread_mutex_lock(&trainingMutex);
#endif
    bool trainingError(false);
    
    if (!this->trainingSet || this->trainingSet->is_empty())
        trainingError = true;
    
    if (check_and_cancel_training())
        return;
    if (!trainingError) {
        try {
            this->train_EM_init();
        } catch (std::exception &e) {
            trainingError = true;
        }
    }
    
    trainingLogLikelihood = -std::numeric_limits<double>::max();
    trainingNbIterations = 0;
    double old_log_prob = trainingLogLikelihood;
    
    while (!train_EM_hasConverged(trainingNbIterations, trainingLogLikelihood, old_log_prob))
    {
        if (check_and_cancel_training())
            return;
        old_log_prob = trainingLogLikelihood;
        if (!trainingError) {
            try {
                trainingLogLikelihood = this->train_EM_update();
            } catch (std::exception &e) {
                trainingError = true;
            }
        }
        
        if (std::isnan(100.*fabs((trainingLogLikelihood-old_log_prob)/old_log_prob)) && (trainingNbIterations > 1))
            trainingError = true;
        
        if (trainingError) {
#ifdef USE_PTHREAD
            is_training_ = false;
            pthread_mutex_unlock(&trainingMutex);
            if (this->trainingCallbackFunction_) {
                this->trainingCallbackFunction_(this, TRAINING_ERROR, this->trainingExtradata_);
            }
#else
            if (this->trainingCallbackFunction_) {
                this->trainingCallbackFunction_(this, TRAINING_ERROR, this->trainingExtradata_);
            }
#endif
            return;
        }
        
        ++trainingNbIterations;
        
        if (stopcriterion.maxSteps > stopcriterion.minSteps)
            this->trainingProgression = float(trainingNbIterations) / float(stopcriterion.maxSteps);
        else
            this->trainingProgression = float(trainingNbIterations) / float(stopcriterion.minSteps);
        
#ifdef USE_PTHREAD
        if (this->trainingCallbackFunction_) {
            this->trainingCallbackFunction_(this, TRAINING_RUN, this->trainingExtradata_);
        }
#else
        if (this->trainingCallbackFunction_) {
            this->trainingCallbackFunction_(this, TRAINING_RUN, this->trainingExtradata_);
        }
#endif
    }
    
    if (check_and_cancel_training())
        return;
    this->train_EM_terminate();
    
#ifdef USE_PTHREAD
    pthread_mutex_unlock(&trainingMutex);
#endif
}

bool xmm::ProbabilisticModel::train_EM_hasConverged(int step, double log_prob, double old_log_prob) const
{
    if (step >= DEFAULT_EMSTOP_ABSOLUTEMAXSTEPS)
        return true;
    if (stopcriterion.maxSteps >= stopcriterion.minSteps)
        return (step >= stopcriterion.maxSteps);
    else
        return (step >= stopcriterion.minSteps) && (100.*fabs((log_prob - old_log_prob) / log_prob) <= stopcriterion.percentChg);
}

void xmm::ProbabilisticModel::train_EM_terminate()
{
    this->trained = true;
    this->trainingSet->set_unchanged();
    
#ifdef USE_PTHREAD
    this->is_training_ = false;
    if (trainingCallbackFunction_) {
        trainingCallbackFunction_(this, TRAINING_DONE, trainingExtradata_);
    }
#else
    if (trainingCallbackFunction_) {
        trainingCallbackFunction_(this, TRAINING_DONE, trainingExtradata_);
    }
#endif
}

#ifdef USE_PTHREAD
bool xmm::ProbabilisticModel::abortTraining(pthread_t this_thread)
{
    if (is_training()) {
        cancel_training_ = true;
        return true;
    }
    return false;
}
#endif

bool xmm::ProbabilisticModel::check_and_cancel_training()
{
#ifdef USE_PTHREAD
    if (!cancel_training_)
        return false;
    pthread_mutex_unlock(&trainingMutex);
    if (this->trainingCallbackFunction_) {
        this->trainingCallbackFunction_(this, TRAINING_ABORT, this->trainingExtradata_);
    }
    is_training_ = false;
    return true;
#else
    return false;
#endif
}

void xmm::ProbabilisticModel::set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata)
{
    prevent_attribute_change();
    trainingExtradata_ = extradata;
    trainingCallbackFunction_ = callback;
}

#pragma mark -
#pragma mark Likelihood Buffer
void xmm::ProbabilisticModel::updateLikelihoodBuffer(double instantLikelihood)
{
    check_training();
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
void xmm::ProbabilisticModel::performance_init()
{
    check_training();
    if (!this->trained)
        throw std::runtime_error("Cannot play: model has not been trained");
    likelihoodBuffer_.clear();
    if (bimodal_) {
        results_predicted_output.resize(dimension_ - dimension_input_);
        results_output_variance.resize(dimension_ - dimension_input_);
    }
}

#pragma mark -
#pragma mark File IO
JSONNode xmm::ProbabilisticModel::to_json() const
{
    check_training();
    JSONNode json_model(JSON_NODE);
    json_model.set_name("ProbabilisticModel");
    
    json_model.push_back(JSONNode("flags", flags_));
    json_model.push_back(JSONNode("bimodal", bimodal_));
    json_model.push_back(JSONNode("dimension", dimension_));
    if (bimodal_)
        json_model.push_back(JSONNode("dimension_input", dimension_input_));
    json_model.push_back(vector2json(column_names_, "column_names"));
    JSONNode json_stopcriterion(JSON_NODE);
    json_stopcriterion.set_name("EMStopCriterion");
    json_stopcriterion.push_back(JSONNode("minsteps", stopcriterion.minSteps));
    json_stopcriterion.push_back(JSONNode("maxsteps", stopcriterion.maxSteps));
    json_stopcriterion.push_back(JSONNode("percentchg", stopcriterion.percentChg));
    json_model.push_back(json_stopcriterion);
    json_model.push_back(JSONNode("likelihoodwindow", likelihoodBuffer_.size()));
    
    return json_model;
}

void xmm::ProbabilisticModel::from_json(JSONNode root)
{
    check_training();
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Flags
        root_it = root.find("flags");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'flags': was expecting 'JSON_NUMBER'", root_it->name());
        if (this->flags_ != static_cast<xmm_flags>(root_it->as_int())) {
            throw JSONException("The flags of the model to read does not match the flags the current instance.", root.name());
        }
        
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
        dimension_ = static_cast<unsigned int>(root_it->as_int());
        
        // Get Input Dimension if bimodal
        if (bimodal_){
            root_it = root.find("dimension_input");
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'dimension_input': was expecting 'JSON_NUMBER'", root_it->name());
            dimension_input_ = static_cast<unsigned int>(root_it->as_int());
        }
        
        // Allocate Memory
        this->allocate();
        
        // Get Column Names
        column_names_.assign(dimension_, "");
        root_it = root.find("column_names");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_ARRAY)
                throw JSONException("Wrong type for node 'column_names': was expecting 'JSON_ARRAY'", root_it->name());
            json2vector(*root_it, column_names_, dimension_);
        }
        
        // Get EM Algorithm stop criterion
        root_it = root.find("EMStopCriterion");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NODE)
            throw JSONException("Wrong type for node 'EMStopCriterion': was expecting 'JSON_NODE'", root_it->name());
        JSONNode json_stopcriterion = *root_it;
        JSONNode::const_iterator crit_it = json_stopcriterion.begin();
        
        if (crit_it == json_stopcriterion.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "minsteps")
            throw JSONException("Wrong name: was expecting 'minsteps'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        stopcriterion.minSteps = static_cast<unsigned int>(crit_it->as_int());
        crit_it++;
        
        if (crit_it == json_stopcriterion.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "maxsteps")
            throw JSONException("Wrong name: was expecting 'maxsteps'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        stopcriterion.maxSteps = static_cast<unsigned int>(crit_it->as_int());
        crit_it++;
        
        if (crit_it == json_stopcriterion.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "percentchg")
            throw JSONException("Wrong name: was expecting 'percentchg'", crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", crit_it->name());
        stopcriterion.percentChg = static_cast<double>(crit_it->as_float());
        
        // Get likelihood window size
        root_it = root.find("likelihoodwindow");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'likelihoodwindow': was expecting 'JSON_NUMBER'", root_it->name());
            this->set_likelihoodwindow(static_cast<unsigned int>(root_it->as_int()));
        }
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (std::exception &e) {
        throw JSONException(e, root.name());
    }
}