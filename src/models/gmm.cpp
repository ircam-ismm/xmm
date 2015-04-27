/*
 * gmm.cpp
 *
 * Gaussian Mixture Model for continuous recognition and regression
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

#include "gmm.h"
#include "kmeans.h"

#pragma mark > Constructors and Utilities
GMM::GMM(rtml_flags flags,
         TrainingSet *trainingSet,
         int nbMixtureComponents,
         double varianceOffset_relative,
         double varianceOffset_absolute,
         GaussianDistribution::COVARIANCE_MODE covariance_mode)
: ProbabilisticModel(flags, trainingSet),
  nbMixtureComponents_(nbMixtureComponents),
  varianceOffset_relative_(varianceOffset_relative),
  varianceOffset_absolute_(varianceOffset_absolute),
  covariance_mode_(covariance_mode)
{
    allocate();
    initParametersToDefault();
}


GMM::GMM(GMM const& src) : ProbabilisticModel(src)
{
    _copy(this, src);
}

GMM& GMM::operator=(GMM const& src)
{
    if(this != &src)
    {
        _copy(this, src);
    }
    return *this;
};

GMM::~GMM()
{
    components.clear();
    mixtureCoeffs.clear();
    beta.clear();
}

#pragma mark > Accessors & Attributes
int GMM::get_nbMixtureComponents() const
{
    return nbMixtureComponents_;
}

double GMM::get_varianceOffset_relative() const
{
    return varianceOffset_relative_;
}

double GMM::get_varianceOffset_absolute() const
{
    return varianceOffset_absolute_;
}

void GMM::set_nbMixtureComponents(int nbMixtureComponents)
{
    PREVENT_ATTR_CHANGE();
    if (nbMixtureComponents < 1) throw invalid_argument("Number of mixture components must be > 0");
    if (nbMixtureComponents == nbMixtureComponents_) return;
    
    nbMixtureComponents_ = nbMixtureComponents;
    allocate();
    this->trained = false;
}

void GMM::set_varianceOffset(double varianceOffset_relative, double varianceOffset_absolute)
{
    PREVENT_ATTR_CHANGE();
    if (varianceOffset_relative <= 0. || varianceOffset_absolute <= 0.)
        throw invalid_argument("Variance offsets must be > 0");
    
    varianceOffset_relative_ = varianceOffset_relative;
    varianceOffset_absolute_ = varianceOffset_absolute;
    for (mixture_iterator component = components.begin() ; component != components.end(); ++component) {
        component->offset_relative = varianceOffset_relative_;
        component->offset_absolute = varianceOffset_absolute_;
    }
}

GaussianDistribution::COVARIANCE_MODE GMM::get_covariance_mode() const
{
    return covariance_mode_;
}

void GMM::set_covariance_mode(GaussianDistribution::COVARIANCE_MODE covariance_mode)
{
    if (covariance_mode == covariance_mode_) return;
    covariance_mode_ = covariance_mode;
    for (unsigned int c=0; c<nbMixtureComponents_; ++c) {
        components[c].set_covariance_mode(covariance_mode);
    }
}

#pragma mark > Performance
void GMM::performance_init()
{
}

double GMM::performance_update(vector<float> const& observation)
{
    double instantaneous_likelihood = likelihood(observation);
    if (bimodal_)
    {
        regression(observation, results_predicted_output);
    }
    return instantaneous_likelihood;
}

#pragma mark > Training
void GMM::train_EM_init()
{
    initParametersToDefault();
    initMeansWithKMeans();
    initCovariances_fullyObserved();
    addCovarianceOffset();
    updateInverseCovariances();
}

#pragma mark > JSON I/O
JSONNode GMM::to_json() const
{
    JSONNode json_gmm(JSON_NODE);
    json_gmm.set_name("GMM");
    
    // Write Parent: EM Learning Model
    JSONNode json_emmodel = ProbabilisticModel::to_json();
    json_emmodel.set_name("ProbabilisticModel");
    json_gmm.push_back(json_emmodel);
    
    // Scalar Attributes
    json_gmm.push_back(JSONNode("nbmixturecomponents", nbMixtureComponents_));
    json_gmm.push_back(JSONNode("varianceoffset_relative", varianceOffset_relative_));
    json_gmm.push_back(JSONNode("varianceoffset_absolute", varianceOffset_absolute_));
    json_gmm.push_back(JSONNode("covariance_mode", covariance_mode_));
    
    // Model Parameters
    json_gmm.push_back(vector2json(mixtureCoeffs, "mixturecoefficients"));
    
    // Mixture Components
    JSONNode json_components(JSON_ARRAY);
    for (int c=0 ; c<nbMixtureComponents_ ; c++)
    {
        json_components.push_back(components[c].to_json());
    }
    json_components.set_name("components");
    json_gmm.push_back(json_components);
    
    return json_gmm;
}

void GMM::from_json(JSONNode root)
{
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::iterator root_it = root.begin();
        
        // Get Parent: ProbabilisticModel
        root_it = root.find("ProbabilisticModel");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NODE)
            throw JSONException("Wrong type for node 'ProbabilisticModel': was expecting 'JSON_NODE'", root_it->name());
        ProbabilisticModel::from_json(*root_it);
        
        // Get Number of Mixture Components
        root_it = root.find("nbmixturecomponents");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'nbmixturecomponents': was expecting 'JSON_NUMBER'", root_it->name());
        nbMixtureComponents_ = static_cast<int>(root_it->as_int());
        
        // Get Covariance Offset (Relative to data variance)
        root_it = root.find("varianceoffset_relative");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'varianceoffset_relative': was expecting 'JSON_NUMBER'", root_it->name());
        varianceOffset_relative_ = root_it->as_float();
        
        // Get Covariance Offset (Minimum value)
        root_it = root.find("varianceoffset_absolute");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'varianceoffset_absolute': was expecting 'JSON_NUMBER'", root_it->name());
        varianceOffset_absolute_ = root_it->as_float();
        
        // Get Covariance mode
        root_it = root.find("covariance_mode");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'covariance_mode': was expecting 'JSON_NUMBER'", root_it->name());
            covariance_mode_ = static_cast<GaussianDistribution::COVARIANCE_MODE>(root_it->as_int());
        } else {
            covariance_mode_ = GaussianDistribution::FULL;
        }
        
        allocate();
        
        // Get Mixture Coefficients
        root_it = root.find("mixturecoefficients");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'mixturecoefficients': was expecting 'JSON_ARRAY'", root_it->name());
        json2vector(*root_it, mixtureCoeffs, nbMixtureComponents_);
        
        // Get Gaussian Mixture Components
        root_it = root.find("components");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'components': was expecting 'JSON_ARRAY'", root_it->name());
        for (int i=0 ; i<nbMixtureComponents_ ; i++) {
            components[i].from_json((*root_it)[i]);
        }
        
        updateInverseCovariances();
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
    
    this->trained = true;
}


#pragma mark > Conversion & Extraction
void GMM::make_bimodal(unsigned int dimension_input)
{
    if (is_training())
        throw runtime_error("Cannot convert model during Training");
    if (bimodal_)
        throw runtime_error("The model is already bimodal");
    if (dimension_input >= dimension_)
        throw out_of_range("Request input dimension exceeds the current dimension");
    set_trainingSet(NULL);
    flags_ = BIMODAL;
    bimodal_ = true;
    dimension_input_ = dimension_input;
    for (mixture_iterator component = components.begin() ; component != components.end(); ++component) {
        component->make_bimodal(dimension_input);
    }
    results_predicted_output.resize(dimension_ - dimension_input_);
    results_output_variance.resize(dimension_ - dimension_input_);
}

void GMM::make_unimodal()
{
    if (is_training())
        throw runtime_error("Cannot convert model during Training");
    if (!bimodal_)
        throw runtime_error("The model is already unimodal");
    set_trainingSet(NULL);
    flags_ = NONE;
    bimodal_ = false;
    dimension_input_ = 0;
    for (mixture_iterator component = components.begin() ; component != components.end(); ++component) {
        component->make_unimodal();
    }
    results_predicted_output.clear();
    results_output_variance.clear();
}

GMM GMM::extract_submodel(vector<unsigned int>& columns) const
{
    if (is_training())
        throw runtime_error("Cannot extract model during Training");
    if (columns.size() > dimension_)
        throw out_of_range("requested number of columns exceeds the dimension of the current model");
    for (unsigned int column=0; column<columns.size(); ++column) {
        if (columns[column] >= dimension_)
            throw out_of_range("Some column indices exceeds the dimension of the current model");
    }
    size_t new_dim =columns.size();
    GMM target_model(*this);
    target_model.set_trainingSet(NULL);
    target_model.set_trainingCallback(NULL, NULL);
    target_model.bimodal_ = false;
    target_model.dimension_ = static_cast<unsigned int>(new_dim);
    target_model.dimension_input_ = 0;
    target_model.flags_ = NONE;
    target_model.allocate();
    target_model.column_names_.resize(new_dim);
    for (unsigned int new_index=0; new_index<new_dim; ++new_index) {
        target_model.column_names_[new_index] = column_names_[columns[new_index]];
    }
    for (unsigned int c=0; c<nbMixtureComponents_; ++c) {
        target_model.components[c] = components[c].extract_submodel(columns);
    }
    target_model.results_predicted_output.clear();
    target_model.results_output_variance.clear();
    return target_model;
}

GMM GMM::extract_submodel_input() const
{
    if (!bimodal_)
        throw runtime_error("The model needs to be bimodal");
    vector<unsigned int> columns_input(dimension_input_);
    for (unsigned int i=0; i<dimension_input_; ++i) {
        columns_input[i] = i;
    }
    return extract_submodel(columns_input);
}

GMM GMM::extract_submodel_output() const
{
    if (!bimodal_)
        throw runtime_error("The model needs to be bimodal");
    vector<unsigned int> columns_output(dimension_ - dimension_input_);
    for (unsigned int i=dimension_input_; i<dimension_; ++i) {
        columns_output[i-dimension_input_] = i;
    }
    return extract_submodel(columns_output);
}

GMM GMM::extract_inverse_model() const
{
    if (!bimodal_)
        throw runtime_error("The model needs to be bimodal");
    vector<unsigned int> columns(dimension_);
    for (unsigned int i=0; i<dimension_-dimension_input_; ++i) {
        columns[i] = i+dimension_input_;
    }
    for (unsigned int i=dimension_-dimension_input_, j=0; i<dimension_; ++i, ++j) {
        columns[i] = j;
    }
    GMM target_model = extract_submodel(columns);
    target_model.make_bimodal(dimension_-dimension_input_);
    return target_model;
}

#pragma mark > Utilities
void GMM::_copy(GMM *dst, GMM const& src)
{
    ProbabilisticModel::_copy(dst, src);
    dst->nbMixtureComponents_ = src.nbMixtureComponents_;
    dst->varianceOffset_relative_ = src.varianceOffset_relative_;
    dst->varianceOffset_absolute_ = src.varianceOffset_absolute_;
    dst->covariance_mode_ = src.covariance_mode_;
    dst->beta.resize(dst->nbMixtureComponents_);
    dst->mixtureCoeffs = src.mixtureCoeffs;
    dst->components = src.components;
}

void GMM::allocate()
{
    mixtureCoeffs.resize(nbMixtureComponents_);
    beta.resize(nbMixtureComponents_);
    components.assign(nbMixtureComponents_,
                      GaussianDistribution(flags_,
                                           dimension_,
                                           dimension_input_,
                                           varianceOffset_relative_,
                                           varianceOffset_absolute_,
                                           covariance_mode_));
}

double GMM::obsProb(const float* observation, int mixtureComponent)
{
    double p(0.);
    
    if (mixtureComponent < 0) {
        for (mixtureComponent=0; mixtureComponent<nbMixtureComponents_; mixtureComponent++) {
            p += obsProb(observation, mixtureComponent);
        }
    } else {
        if (mixtureComponent >= nbMixtureComponents_)
            throw out_of_range("The index of the Gaussian Mixture Component is out of bounds");
        p = mixtureCoeffs[mixtureComponent] * components[mixtureComponent].likelihood(observation);
    }
    
    return p;
}

double GMM::obsProb_input(const float* observation_input,
                          int mixtureComponent)
{
    if (!bimodal_)
        throw runtime_error("Model is not bimodal. Use the function 'obsProb'");
    double p(0.);
    
    if (mixtureComponent < 0) {
        for (mixtureComponent=0; mixtureComponent<nbMixtureComponents_; mixtureComponent++) {
            p += obsProb_input(observation_input, mixtureComponent);
        }
    } else {
        
        p = mixtureCoeffs[mixtureComponent] * components[mixtureComponent].likelihood_input(observation_input);
    }
    
    return p;
}

double GMM::obsProb_bimodal(const float* observation_input,
                            const float* observation_output,
                            int mixtureComponent)
{
    if (!bimodal_)
        throw runtime_error("Model is not bimodal. Use the function 'obsProb'");
    double p(0.);
    
    if (mixtureComponent < 0) {
        for (mixtureComponent=0; mixtureComponent<nbMixtureComponents_; mixtureComponent++) {
            p += obsProb_bimodal(observation_input, observation_output, mixtureComponent);
        }
    } else {
        
        p = mixtureCoeffs[mixtureComponent] * components[mixtureComponent].likelihood_bimodal(observation_input, observation_output);
    }
    
    return p;
}

#pragma mark > Training
void GMM::initMeansWithKMeans()
{
    if (!this->trainingSet || this->trainingSet->is_empty())
        return;
    KMeans* kmeans = new KMeans(this->trainingSet, nbMixtureComponents_);
    kmeans->trainingInitType = KMeans::BIASED;
    kmeans->train();
    for (int c=0; c<nbMixtureComponents_; c++) {
        for (unsigned int d=0; d<dimension_; ++d) {
            components[c].mean[d] = kmeans->centers[c*dimension_+d];
        }
    }
    delete kmeans;
}

void GMM::initCovariances_fullyObserved()
{
    // TODO: simplify with covariance symmetricity
    // TODO: If Kmeans, covariances from cluster members
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    int nbPhrases = this->trainingSet->size();
    
    if (covariance_mode_ == GaussianDistribution::FULL) {
        for (int n=0; n<nbMixtureComponents_; n++)
            components[n].covariance.assign(dimension_*dimension_, 0.0);
    } else {
        for (int n=0; n<nbMixtureComponents_; n++)
            components[n].covariance.assign(dimension_, 0.0);
    }
    
    vector<double> gmeans(nbMixtureComponents_*dimension_, 0.0);
    vector<int> factor(nbMixtureComponents_, 0);
    for (int i=0; i<nbPhrases; i++) {
        int step = ((*this->trainingSet)(i))->second->length() / nbMixtureComponents_;
        int offset(0);
        for (int n=0; n<nbMixtureComponents_; n++) {
            for (int t=0; t<step; t++) {
                for (int d1=0; d1<dimension_; d1++) {
                    gmeans[n*dimension_+d1] += (*((*this->trainingSet)(i)->second))(offset+t, d1);
                    if (covariance_mode_ == GaussianDistribution::FULL) {
                        for (int d2=0; d2<dimension_; d2++) {
                            components[n].covariance[d1*dimension_+d2] += (*((*this->trainingSet)(i)->second))(offset+t, d1) * (*((*this->trainingSet)(i)->second))(offset+t, d2);
                        }
                    } else {
                        float value = (*((*this->trainingSet)(i)->second))(offset+t, d1);
                        components[n].covariance[d1] += value * value;
                    }
                }
            }
            offset += step;
            factor[n] += step;
        }
    }
    
    for (int n=0; n<nbMixtureComponents_; n++) {
        for (int d1=0; d1<dimension_; d1++) {
            gmeans[n*dimension_+d1] /= factor[n];
            if (covariance_mode_ == GaussianDistribution::FULL) {
                for (int d2=0; d2<dimension_; d2++)
                    components[n].covariance[d1*dimension_+d2] /= factor[n];
            } else {
                components[n].covariance[d1] /= factor[n];
            }
        }
    }
    
    for (int n=0; n<nbMixtureComponents_; n++) {
        for (int d1=0; d1<dimension_; d1++) {
            if (covariance_mode_ == GaussianDistribution::FULL) {
                for (int d2=0; d2<dimension_; d2++)
                    components[n].covariance[d1*dimension_+d2] -= gmeans[n*dimension_+d1]*gmeans[n*dimension_+d2];
            } else {
                components[n].covariance[d1] -= gmeans[n*dimension_+d1] * gmeans[n*dimension_+d1];
            }
        }
    }
}

double GMM::train_EM_update()
{
    double log_prob(0.);
    
    int totalLength(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); ++it)
        totalLength += it->second->length();
    
    vector< vector<double> > p(nbMixtureComponents_);
    vector<double> E(nbMixtureComponents_, 0.0);
    for (int c=0; c<nbMixtureComponents_; c++) {
        p[c].resize(totalLength);
        E[c] = 0.;
    }
    
    int tbase(0);
    
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); ++it) {
        int T = it->second->length();
        for (int t=0; t<T; t++) {
            double norm_const(0.);
            for (int c=0; c<nbMixtureComponents_; c++)
            {
                if (bimodal_) {
                    p[c][tbase+t] = obsProb_bimodal(it->second->get_dataPointer_input(t),
                                                    it->second->get_dataPointer_output(t),
                                                    c);
                } else {
                    p[c][tbase+t] = obsProb(it->second->get_dataPointer(t), c);
                }
                
                if (p[c][tbase+t] == 0. || isnan(p[c][tbase+t]) || isinf(p[c][tbase+t])) {
                    p[c][tbase+t] = 1e-100;
                }
                norm_const += p[c][tbase+t];
            }
            for (int c=0; c<nbMixtureComponents_; c++) {
                p[c][tbase+t] /= norm_const;
                E[c] += p[c][tbase+t];
            }
            log_prob += log(norm_const);
        }
        tbase += T;
    }
    
    // Estimate Mixture coefficients
    for (int c=0; c<nbMixtureComponents_; c++) {
        mixtureCoeffs[c] = E[c]/double(totalLength);
    }
    
    // Estimate means
    for (int c=0; c<nbMixtureComponents_; c++) {
        for (int d=0; d<dimension_; d++) {
            components[c].mean[d] = 0.;
            tbase = 0;
            for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); ++it) {
                int T = it->second->length();
                for (int t=0; t<T; t++) {
                    components[c].mean[d] += p[c][tbase+t] * (*it->second)(t, d);
                }
                tbase += T;
            }
            components[c].mean[d] /= E[c];
        }
    }
    
    //estimate covariances
    if (covariance_mode_ == GaussianDistribution::FULL) {
        for (int c=0; c<nbMixtureComponents_; c++) {
            for (int d1=0; d1<dimension_; d1++) {
                for (int d2=d1; d2<dimension_; d2++) {
                    components[c].covariance[d1 * dimension_ + d2] = 0.;
                    tbase = 0;
                    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); ++it) {
                        int T = it->second->length();
                        for (int t=0; t<T; t++) {
                            components[c].covariance[d1 * dimension_ + d2] += p[c][tbase+t]
                            * ((*it->second)(t, d1) - components[c].mean[d1])
                            * ((*it->second)(t, d2) - components[c].mean[d2]);
                        }
                        tbase += T;
                    }
                    components[c].covariance[d1 * dimension_ + d2] /= E[c];
                    if (d1 != d2)
                        components[c].covariance[d2*dimension_+d1] = components[c].covariance[d1*dimension_+d2];
                }
            }
        }
    }
    else
    {
        for (int c=0; c<nbMixtureComponents_; c++) {
            for (int d1=0; d1<dimension_; d1++) {
                components[c].covariance[d1] = 0.;
                tbase = 0;
                for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); ++it) {
                    int T = it->second->length();
                    for (int t=0; t<T; t++) {
                        float value = ((*it->second)(t, d1) - components[c].mean[d1]);
                        components[c].covariance[d1] += p[c][tbase+t] * value * value;
                    }
                    tbase += T;
                }
                components[c].covariance[d1] /= E[c];
            }
        }
    }
    
    addCovarianceOffset();
    updateInverseCovariances();
    
    return log_prob;
}

void GMM::initParametersToDefault()
{
    vector<float> global_trainingdata_var(dimension_, 1.0);
    if (this->trainingSet)
        global_trainingdata_var = this->trainingSet->variance();
    
    double norm_coeffs(0.);
    for (int c=0; c<nbMixtureComponents_; c++) {
        components[c].scale.assign(global_trainingdata_var.begin(), global_trainingdata_var.end());
        if (covariance_mode_ == GaussianDistribution::FULL) {
            components[c].covariance.assign(dimension_*dimension_, varianceOffset_absolute_/2.);
        } else {
            components[c].covariance.assign(dimension_, varianceOffset_absolute_/2.);
        }
        components[c].addOffset();
        mixtureCoeffs[c] = 1./float(nbMixtureComponents_);
        norm_coeffs += mixtureCoeffs[c];
    }
    for (int c=0; c<nbMixtureComponents_; c++) {
        mixtureCoeffs[c] /= norm_coeffs;
    }
}

void GMM::normalizeMixtureCoeffs()
{
    double norm_const(0.);
    for (int c=0; c<nbMixtureComponents_; c++) {
        norm_const += mixtureCoeffs[c];
    }
    if (norm_const > 0) {
        for (int c=0; c<nbMixtureComponents_; c++) {
            mixtureCoeffs[c] /= norm_const;
        }
    } else {
        for (int c=0; c<nbMixtureComponents_; c++) {
            mixtureCoeffs[c] = 1/float(nbMixtureComponents_);
        }
    }
}

void GMM::addCovarianceOffset()
{
    for (mixture_iterator component = components.begin() ; component != components.end(); ++component) {
        component->addOffset();
    }
}

void GMM::updateInverseCovariances()
{
    try {
        for (mixture_iterator component = components.begin() ; component != components.end() ; ++component) {
            component->updateInverseCovariance();
            if (bimodal_)
                component->updateOutputVariances();
        }
    } catch (exception& e) {
        throw runtime_error("Matrix inversion error: varianceoffset must be too small");
    }
}

#pragma mark > Performance
void GMM::regression(vector<float> const& observation_input, vector<float>& predicted_output)
{
    int dimension_output = dimension_ - dimension_input_;
    predicted_output.assign(dimension_output, 0.0);
    results_output_variance.assign(dimension_output, 0.0);
    vector<float> tmp_predicted_output(dimension_output, 0.0);
    
    for (int c=0; c<nbMixtureComponents_; c++) {
        components[c].regression(observation_input, tmp_predicted_output);
        for (int d = 0; d < dimension_output; ++d)
        {
            predicted_output[d] += beta[c] * tmp_predicted_output[d];
            results_output_variance[d] += beta[c] * beta[c] * components[c].output_variance[d];
        }
    }
}

double GMM::likelihood(vector<float> const& observation, vector<float> const& observation_output)
{
    double likelihood(0.);
    for (int c=0; c<nbMixtureComponents_; c++) {
        if (bimodal_) {
            if (observation_output.empty())
                beta[c] = obsProb_input(&observation[0], c);
            else
                beta[c] = obsProb_bimodal(&observation[0], &observation_output[0], c);
        } else {
            beta[c] = obsProb(&observation[0], c);
        }
        likelihood += beta[c];
    }
    for (int c=0; c<nbMixtureComponents_; c++) {
        beta[c] /= likelihood;
    }
    
    this->updateLikelihoodBuffer(likelihood);
    return likelihood;
}
