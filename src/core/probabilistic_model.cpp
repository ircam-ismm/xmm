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
ProbabilisticModel::ProbabilisticModel(rtml_flags flags,
                                       TrainingSet *trainingSet)
: trainingSet(trainingSet),
  trained(false),
  trainingProgression(0),
  flags_(flags),
  bimodal_(flags & BIMODAL),
  trainingCallbackFunction_(NULL),
  is_training_(false)
{
    if (this->trainingSet) {
        this->trainingSet->add_listener(this);
        if (bimodal_ && !trainingSet->is_bimodal())
            throw runtime_error("This model is bimodal but the training set is not. Can't Connect.");
        if (!bimodal_ && trainingSet->is_bimodal())
            throw runtime_error("This model is not bimodal but the training set is. Can't Connect.");
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
    stopcriterion.minSteps = EM_MODEL_DEFAULT_EMSTOP_MINSTEPS;
    stopcriterion.maxSteps = EM_MODEL_DEFAULT_EMSTOP_MAXSTEPS;
    stopcriterion.percentChg = EM_MODEL_DEFAULT_EMSTOP_PERCENT_CHG;
    likelihoodBuffer_.resize(EM_MODEL_DEFAULT_LIKELIHOOD_WINDOW);
#ifdef USE_PTHREAD
    pthread_mutex_init(&trainingMutex, NULL);
#endif
}

ProbabilisticModel::ProbabilisticModel(ProbabilisticModel const& src)
{
    this->_copy(this, src);
}

ProbabilisticModel& ProbabilisticModel::operator=(ProbabilisticModel const& src)
{
    if(this != &src)
    {
        if (this->trainingSet)
            this->trainingSet->remove_listener(this);
        _copy(this, src);
    }
    return *this;
};

void ProbabilisticModel::_copy(ProbabilisticModel *dst,
                               ProbabilisticModel const& src)
{
    if (src.is_training())
        throw runtime_error("Cannot copy model during Training");
    dst->is_training_ = false;
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

ProbabilisticModel::~ProbabilisticModel()
{
    if (this->trainingSet)
        this->trainingSet->remove_listener(this);
}

#pragma mark -
#pragma mark Accessors
bool ProbabilisticModel::is_training() const
{
    return is_training_;
}

void ProbabilisticModel::set_trainingSet(TrainingSet *trainingSet)
{
    PREVENT_ATTR_CHANGE();
    if (this->trainingSet)
        this->trainingSet->remove_listener(this);
    this->trainingSet = trainingSet;
    if (this->trainingSet) {
        this->trainingSet->add_listener(this);
        if (bimodal_ && !trainingSet->is_bimodal())
            throw runtime_error("This model is bimodal but the training set is not. Can't Connect.");
        if (!bimodal_ && trainingSet->is_bimodal())
            throw runtime_error("This model is not bimodal but the training set is. Can't Connect.");
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
    this->allocate();
}

void ProbabilisticModel::notify(string attribute)
{
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

unsigned int ProbabilisticModel::get_likelihoodwindow() const
{
    return likelihoodBuffer_.size();
}

void ProbabilisticModel::set_likelihoodwindow(unsigned int likelihoodwindow)
{
    if (likelihoodwindow < 1) throw invalid_argument("Likelihood Buffer size must be > 1");
    likelihoodBuffer_.resize(likelihoodwindow);
}

vector<string> const& ProbabilisticModel::get_column_names() const
{
    return column_names_;
}

#pragma mark -
#pragma mark Training
void* ProbabilisticModel::train_func(void *context)
{
    ((ProbabilisticModel *)context)->train();
    return NULL;
}

void ProbabilisticModel::train()
{
#ifdef USE_PTHREAD
    pthread_mutex_lock(&trainingMutex);
#endif
    bool trainingError(false);
    
    if (!this->trainingSet || this->trainingSet->is_empty())
        trainingError = true;
    
    is_training_ = true;
    
    if (!trainingError) {
        try {
            this->train_EM_init();
        } catch (exception &e) {
            trainingError = true;
        }
    }
    
    trainingLogLikelihood = log(0.);
    trainingNbIterations = 0;
    double old_log_prob = trainingLogLikelihood;
    
    while (!train_EM_hasConverged(trainingNbIterations, trainingLogLikelihood, old_log_prob))
    {
        old_log_prob = trainingLogLikelihood;
        if (!trainingError) {
            try {
                trainingLogLikelihood = this->train_EM_update();
            } catch (exception &e) {
                trainingError = true;
            }
        }
        
        if (isnan(100.*fabs((trainingLogLikelihood-old_log_prob)/old_log_prob)) && (trainingNbIterations > 1))
            trainingError = true;
        
        if (trainingError) {
#ifdef USE_PTHREAD
            pthread_mutex_unlock(&trainingMutex);
            if (this->trainingCallbackFunction_) {
                pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
                this->trainingCallbackFunction_(this, TRAINING_ERROR, this->trainingExtradata_);
                pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
                pthread_testcancel();
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
            pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
            this->trainingCallbackFunction_(this, TRAINING_RUN, this->trainingExtradata_);
            pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
            pthread_testcancel();
        }
#else
        if (this->trainingCallbackFunction_) {
            this->trainingCallbackFunction_(this, TRAINING_RUN, this->trainingExtradata_);
        }
#endif
    }
    
    this->train_EM_terminate();
    
#ifdef USE_PTHREAD
    pthread_mutex_unlock(&trainingMutex);
#endif
}

bool ProbabilisticModel::train_EM_hasConverged(int step, double log_prob, double old_log_prob) const
{
    if (step >= EM_MODEL_DEFAULT_EMSTOP_ABSOLUTEMAXSTEPS)
        return true;
    if (stopcriterion.maxSteps >= stopcriterion.minSteps)
        return (step >= stopcriterion.maxSteps);
    else
        return (step >= stopcriterion.minSteps) && (100.*fabs((log_prob - old_log_prob) / log_prob) <= stopcriterion.percentChg);
}

void ProbabilisticModel::train_EM_terminate()
{
    this->trained = true;
    this->trainingSet->set_unchanged();
    this->is_training_ = false;
    
#ifdef USE_PTHREAD
    if (trainingCallbackFunction_) {
        pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
        trainingCallbackFunction_(this, TRAINING_DONE, trainingExtradata_);
        pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
        pthread_testcancel();
    }
#else
    if (trainingCallbackFunction_) {
        trainingCallbackFunction_(this, TRAINING_DONE, trainingExtradata_);
    }
#endif
}

#ifdef USE_PTHREAD
void ProbabilisticModel::abortTraining(pthread_t this_thread)
{
    if (!is_training_)
        return;
    pthread_cancel(this_thread);
    void *status;
    pthread_join(this_thread, &status);
    pthread_mutex_unlock(&trainingMutex);
    is_training_ = false;
    if (trainingCallbackFunction_) {
        trainingCallbackFunction_(this, TRAINING_ABORT, trainingExtradata_);
    }
}
#endif

void ProbabilisticModel::set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata)
{
    PREVENT_ATTR_CHANGE();
    trainingExtradata_ = extradata;
    trainingCallbackFunction_ = callback;
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
    if (bimodal_) {
        results_predicted_output.resize(dimension_ - dimension_input_);
        results_output_variance.resize(dimension_ - dimension_input_);
    }
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
        
        // Get Column Names
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "column_names")
            throw JSONException("Wrong name: was expecting 'column_names'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        column_names_.resize(dimension_);
        json2vector(*root_it, column_names_, dimension_);
        ++root_it;
        
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