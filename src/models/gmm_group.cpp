/*
 * gmm_group.cpp
 *
 * Group of Gaussian Mixture Models for continuous recognition and regression with multiple classes
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

#include "gmm_group.h"

#pragma mark -
#pragma mark Constructor
GMMGroup::GMMGroup(rtml_flags flags,
                   TrainingSet *_globalTrainingSet)
: ModelGroup< GMM >(flags, _globalTrainingSet)
{
    bimodal_ = (flags & BIMODAL);
}

#pragma mark -
#pragma mark Get & Set
int GMMGroup::get_nbMixtureComponents() const
{
    return this->referenceModel_.get_nbMixtureComponents();
}

void GMMGroup::set_nbMixtureComponents(int nbMixtureComponents_)
{
    PREVENT_ATTR_CHANGE();
    this->referenceModel_.set_nbMixtureComponents(nbMixtureComponents_);
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.set_nbMixtureComponents(nbMixtureComponents_);
    }
}

double GMMGroup::get_varianceOffset_relative() const
{
    return this->referenceModel_.get_varianceOffset_relative();
}

double GMMGroup::get_varianceOffset_absolute() const
{
    return this->referenceModel_.get_varianceOffset_absolute();
}

void GMMGroup::set_varianceOffset(double varianceOffset_relative, double varianceOffset_absolute)
{
    PREVENT_ATTR_CHANGE();
    this->referenceModel_.set_varianceOffset(varianceOffset_relative, varianceOffset_absolute);
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.set_varianceOffset(varianceOffset_relative, varianceOffset_absolute);
    }
}

#pragma mark -
#pragma mark Performance
void GMMGroup::performance_update(vector<float> const& observation)
{
    int i(0);
    for (model_iterator it = this->models.begin(); it != this->models.end(); ++it) {
        results_instant_likelihoods[i] = it->second.performance_update(observation);
    }
    
    update_likelihood_results();
    
    if (bimodal_) {
        unsigned int dimension = this->referenceModel_.dimension();
        unsigned int dimension_input = this->referenceModel_.dimension_input();
        unsigned int dimension_output = dimension - dimension_input;
        
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
            for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
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
JSONNode GMMGroup::to_json() const
{
    JSONNode json_ccmodels(JSON_NODE);
    json_ccmodels.set_name("GMMGroup");
    json_ccmodels.push_back(JSONNode("bimodal", bimodal_));
    json_ccmodels.push_back(JSONNode("dimension", dimension()));
    if (bimodal_)
        json_ccmodels.push_back(JSONNode("dimension_input", dimension_input()));
    json_ccmodels.push_back(JSONNode("size", models.size()));
    json_ccmodels.push_back(JSONNode("performancemode", int(performanceMode_)));
    json_ccmodels.push_back(JSONNode("nbmixturecomponents", get_nbMixtureComponents()));
    json_ccmodels.push_back(JSONNode("varianceoffset_relative", get_varianceOffset_relative()));
    json_ccmodels.push_back(JSONNode("varianceoffset_absolute", get_varianceOffset_absolute()));
    
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
    json_ccmodels.push_back(json_models);
    
    return json_ccmodels;
}

void GMMGroup::from_json(JSONNode root)
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
        
        // Get Mixture Components
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
            if (array_it->name() != "GMM")
                throw JSONException("Wrong name: was expecting 'GMM'", array_it->name());
            if (array_it->type() != JSON_NODE)
                throw JSONException("Wrong type: was expecting 'JSON_NODE'", array_it->name());
            models[l] = this->referenceModel_;
            models[l].from_json(*array_it);
        }
        
        if (numModels != models.size())
            throw JSONException("Number of models does not match", root.name());
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}
