//
// learning_model.cpp
//
// Base class for machine learning models. Real-time implementation and interface.
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#include "base_model.h"

#pragma mark -
#pragma mark Constructors
BaseModel::BaseModel(rtml_flags flags,
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
}

BaseModel::BaseModel(BaseModel const& src)
{
    this->_copy(this, src);
}

BaseModel& BaseModel::operator=(BaseModel const& src)
{
    if(this != &src)
    {
        _copy(this, src);
    }
    return *this;
}

void BaseModel::_copy(BaseModel *dst, BaseModel const& src)
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
}

BaseModel::~BaseModel()
{}

#pragma mark -
#pragma mark Connect Training set
void BaseModel::set_trainingSet(TrainingSet *trainingSet)
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

void BaseModel::notify(string attribute)
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

#pragma mark -
#pragma mark Callback function for training
void BaseModel::set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata) {
    trainingExtradata_ = extradata;
    trainingCallback_ = callback;
}

#pragma mark -
#pragma mark Pure Virtual Methods: Training, Playing
void BaseModel::finishTraining()
{
    if (trainingCallback_)
        trainingCallback_(this, TRAINING_DONE, trainingExtradata_);
}
void BaseModel::initPlaying()
{
    if (!this->trained) {
        throw runtime_error("Cannot play: model has not been trained");
    }
}

#pragma mark -
#pragma mark File IO
JSONNode BaseModel::to_json() const
{
    JSONNode json_model(JSON_NODE);
    json_model.set_name("BaseModel");
    json_model.push_back(JSONNode("flags", flags_));
    json_model.push_back(JSONNode("bimodal", bimodal_));
    json_model.push_back(JSONNode("dimension", dimension_));
    if (bimodal_)
        json_model.push_back(JSONNode("dimension_input", dimension_input_));
    
    return json_model;
}

void BaseModel::from_json(JSONNode root)
{
    try {
        assert(root.type() == JSON_NODE);
        JSONNode::const_iterator root_it = root.begin();
        
        // Get Flags
        assert(root_it != root.end());
        assert(root_it->name() == "flags");
        assert(root_it->type() == JSON_NUMBER);
        if(this->flags_ != rtml_flags(root_it->as_int())) {
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
        dimension_ = root_it->as_int();
        ++root_it;
        
        // Get Input Dimension if bimodal
        if (bimodal_){
            assert(root_it != root.end());
            assert(root_it->name() == "dimension_input");
            assert(root_it->type() == JSON_NUMBER);
            dimension_input_ = root_it->as_int();
            ++root_it;
        }
        
        this->allocate();
        
    } catch (JSONException &e) {
        throw JSONException(e);
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}