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
xmm::GMMGroup::GMMGroup(xmm_flags flags,
                   TrainingSet *_globalTrainingSet,
                   GaussianDistribution::COVARIANCE_MODE covariance_mode)
: ModelGroup< GMM >(flags, _globalTrainingSet)
{
    bimodal_ = (flags & BIMODAL);
    set_covariance_mode(covariance_mode);
}

xmm::GMMGroup::GMMGroup(GMMGroup const& src)
{
    this->_copy(this, src);
}

xmm::GMMGroup& xmm::GMMGroup::operator=(GMMGroup const& src)
{
    if(this != &src)
    {
        if (this->globalTrainingSet)
            this->globalTrainingSet->remove_listener(this);
        _copy(this, src);
    }
    return *this;
}

void xmm::GMMGroup::_copy(GMMGroup *dst, GMMGroup const& src)
{
    ModelGroup<GMM>::_copy(dst, src);
}


#pragma mark -
#pragma mark Get & Set
int xmm::GMMGroup::get_nbMixtureComponents() const
{
    return this->referenceModel_.get_nbMixtureComponents();
}

void xmm::GMMGroup::set_nbMixtureComponents(int nbMixtureComponents_)
{
    prevent_attribute_change();
    this->referenceModel_.set_nbMixtureComponents(nbMixtureComponents_);
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.set_nbMixtureComponents(nbMixtureComponents_);
    }
}

double xmm::GMMGroup::get_varianceOffset_relative() const
{
    return this->referenceModel_.get_varianceOffset_relative();
}

double xmm::GMMGroup::get_varianceOffset_absolute() const
{
    return this->referenceModel_.get_varianceOffset_absolute();
}

void xmm::GMMGroup::set_varianceOffset(double varianceOffset_relative, double varianceOffset_absolute)
{
    prevent_attribute_change();
    this->referenceModel_.set_varianceOffset(varianceOffset_relative, varianceOffset_absolute);
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.set_varianceOffset(varianceOffset_relative, varianceOffset_absolute);
    }
}

xmm::GaussianDistribution::COVARIANCE_MODE xmm::GMMGroup::get_covariance_mode() const
{
    return this->referenceModel_.get_covariance_mode();
}

void xmm::GMMGroup::set_covariance_mode(GaussianDistribution::COVARIANCE_MODE covariance_mode)
{
    prevent_attribute_change();
    if (covariance_mode == get_covariance_mode()) return;
    try {
        this->referenceModel_.set_covariance_mode(covariance_mode);
    } catch (std::exception& e) {
        if (strncmp(e.what(), "Non-invertible matrix", 21) != 0)
            throw std::runtime_error(e.what());
    }
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.set_covariance_mode(covariance_mode);
    }
}

#pragma mark -
#pragma mark Performance
void xmm::GMMGroup::performance_update(std::vector<float> const& observation)
{
    check_training();
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
JSONNode xmm::GMMGroup::to_json() const
{
    check_training();
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
    json_ccmodels.push_back(JSONNode("covariance_mode", get_covariance_mode()));
    
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

void xmm::GMMGroup::from_json(JSONNode root)
{
    check_training();
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
        
        // Get Mixture Components
        root_it = root.find("nbmixturecomponents");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'nbmixturecomponents': was expecting 'JSON_NUMBER'", root_it->name());
        set_nbMixtureComponents(static_cast<unsigned int>(root_it->as_int()));
        
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
            
            // Get GMM
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
    } catch (std::exception &e) {
        throw JSONException(e, root.name());
    }
}

#pragma mark > Conversion & Extraction
void xmm::GMMGroup::make_bimodal(unsigned int dimension_input)
{
    check_training();
    if (bimodal_)
        throw std::runtime_error("The model is already bimodal");
    if (dimension_input >= dimension())
        throw std::out_of_range("Request input dimension exceeds the current dimension");
    
    try {
        this->referenceModel_.make_bimodal(dimension_input);
    } catch (std::exception const& e) {
    }
    bimodal_ = true;
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.make_bimodal(dimension_input);
    }
    set_trainingSet(NULL);
    results_predicted_output.resize(dimension() - this->dimension_input());
    results_output_variance.resize(dimension() - this->dimension_input());
}

void xmm::GMMGroup::make_unimodal()
{
    check_training();
    if (!bimodal_)
        throw std::runtime_error("The model is already unimodal");
    this->referenceModel_.make_unimodal();
    for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
        it->second.make_unimodal();
    }
    set_trainingSet(NULL);
    results_predicted_output.clear();
    results_output_variance.clear();
    bimodal_ = false;
}

xmm::GMMGroup xmm::GMMGroup::extract_submodel(std::vector<unsigned int>& columns) const
{
    check_training();
    if (columns.size() > this->dimension())
        throw std::out_of_range("requested number of columns exceeds the dimension of the current model");
    for (unsigned int column=0; column<columns.size(); ++column) {
        if (columns[column] >= this->dimension())
            throw std::out_of_range("Some column indices exceeds the dimension of the current model");
    }
    GMMGroup target_model(*this);
    target_model.set_trainingSet(NULL);
    target_model.bimodal_ = false;
    target_model.referenceModel_ = this->referenceModel_.extract_submodel(columns);
    for (model_iterator it=target_model.models.begin(); it != target_model.models.end(); ++it) {
        it->second = this->models.at(it->first).extract_submodel(columns);
    }
    target_model.results_predicted_output.clear();
    target_model.results_output_variance.clear();
    return target_model;
}

xmm::GMMGroup xmm::GMMGroup::extract_submodel_input() const
{
    check_training();
    if (!bimodal_)
        throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns_input(dimension_input());
    for (unsigned int i=0; i<dimension_input(); ++i) {
        columns_input[i] = i;
    }
    return extract_submodel(columns_input);
}

xmm::GMMGroup xmm::GMMGroup::extract_submodel_output() const
{
    check_training();
    if (!bimodal_)
        throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns_output(dimension() - dimension_input());
    for (unsigned int i=dimension_input(); i<dimension(); ++i) {
        columns_output[i-dimension_input()] = i;
    }
    return extract_submodel(columns_output);
}

xmm::GMMGroup xmm::GMMGroup::extract_inverse_model() const
{
    check_training();
    if (!bimodal_)
        throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns(dimension());
    for (unsigned int i=0; i<dimension()-dimension_input(); ++i) {
        columns[i] = i+dimension_input();
    }
    for (unsigned int i=dimension()-dimension_input(), j=0; i<dimension(); ++i, ++j) {
        columns[i] = j;
    }
    GMMGroup target_model = extract_submodel(columns);
    target_model.make_bimodal(dimension()-dimension_input());
    return target_model;
}
