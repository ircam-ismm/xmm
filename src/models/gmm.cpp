//
// gmm.cpp
//
// Gaussian Mixture Model
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
//

#include "gmm.h"
#include "kmeans.h"

#pragma mark > Constructors and Utilities
GMM::GMM(rtml_flags flags,
         TrainingSet *trainingSet,
         int nbMixtureComponents,
         double varianceOffset_relative,
         double varianceOffset_absolute)
: ProbabilisticModel(flags, trainingSet)
{
    nbMixtureComponents_ = nbMixtureComponents;
    varianceOffset_relative_ = varianceOffset_relative;
    varianceOffset_absolute_ = varianceOffset_absolute;
    weight_regression_ = 1.;
    
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

double GMM::get_weight_regression() const
{
    return weight_regression_;
}

void GMM::set_weight_regression(double weight_regression)
{
    weight_regression_ = weight_regression;
    for (mixture_iterator component = components.begin() ; component != components.end(); ++component) {
        component->weight_regression = weight_regression_;
    }
}

#pragma mark > Performance
void GMM::performance_init()
{
    if (bimodal_)
        results_predicted_output.resize(dimension_ - dimension_input_);
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
    if (this->trainingSet->size() > 1) {
        initMeansWithKMeans();
    } else {
        initMeansWithFirstPhrase();
    }
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
    json_gmm.push_back(JSONNode("weight_regression", weight_regression_));
    
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
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "ProbabilisticModel")
            throw JSONException("Wrong name: was expecting 'ProbabilisticModel'", root_it->name());
        if (root_it->type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root_it->name());
        ProbabilisticModel::from_json(*root_it);
        ++root_it;
        
        // Get Mixture Components
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "nbmixturecomponents")
            throw JSONException("Wrong name: was expecting 'nbmixturecomponents'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        nbMixtureComponents_ = root_it->as_int();
        ++root_it;
        
        // Get Covariance Offset (Relative)
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "varianceoffset_relative")
            throw JSONException("Wrong name: was expecting 'varianceoffset_relative'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        varianceOffset_relative_ = root_it->as_float();
        ++root_it;
        
        // Get Covariance Offset (Absolute)
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "varianceoffset_absolute")
            throw JSONException("Wrong name: was expecting 'varianceoffset_absolute'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        varianceOffset_absolute_ = root_it->as_float();
        ++root_it;
        
        allocate();
        
        // Get Regresion Weight
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "weight_regression")
            throw JSONException("Wrong name: was expecting 'weight_regression'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        weight_regression_ = root_it->as_float();
        ++root_it;
        
        // Get Mixture Coefficients
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "mixturecoefficients")
            throw JSONException("Wrong name: was expecting 'mixturecoefficients'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        json2vector(*root_it, mixtureCoeffs, nbMixtureComponents_);
        ++root_it;
        
        // Get Gaussian Mixture Components
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "components")
            throw JSONException("Wrong name: was expecting 'components'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
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

#pragma mark > Utilities
void GMM::_copy(GMM *dst, GMM const& src)
{
    ProbabilisticModel::_copy(dst, src);
    dst->nbMixtureComponents_ = src.nbMixtureComponents_;
    dst->varianceOffset_relative_ = src.varianceOffset_relative_;
    dst->varianceOffset_absolute_ = src.varianceOffset_absolute_;
    dst->weight_regression_ = src.weight_regression_;
    dst->mixtureCoeffs = src.mixtureCoeffs;
    dst->components = src.components;
    
    dst->allocate();
}

void GMM::allocate()
{
    mixtureCoeffs.resize(nbMixtureComponents_);
    beta.resize(nbMixtureComponents_);
    components.assign(nbMixtureComponents_, GaussianDistribution(flags_, dimension_, dimension_input_, varianceOffset_relative_, varianceOffset_absolute_));
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
void GMM::initMeansWithFirstPhrase()
{
    if (!this->trainingSet || this->trainingSet->is_empty())
        return;
    int step = this->trainingSet->begin()->second->length() / nbMixtureComponents_;
    
    int offset(0);
    for (int c=0; c<nbMixtureComponents_; c++) {
        for (int d=0; d<dimension_; d++) {
            components[c].mean[d] = 0.0;
        }
        for (int t=0; t<step; t++) {
            for (int d=0; d<dimension_; d++) {
                components[c].mean[d] += (*this->trainingSet->begin()->second)(offset+t, d) / float(step);
            }
        }
        offset += step;
    }
}

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
    
    for (int n=0; n<nbMixtureComponents_; n++)
        components[n].covariance.assign(dimension_*dimension_, 0.0);
    
    vector<double> gmeans(nbMixtureComponents_*dimension_, 0.0);
    vector<int> factor(nbMixtureComponents_, 0);
    for (int i=0; i<nbPhrases; i++) {
        int step = ((*this->trainingSet)(i))->second->length() / nbMixtureComponents_;
        int offset(0);
        for (int n=0; n<nbMixtureComponents_; n++) {
            for (int t=0; t<step; t++) {
                for (int d1=0; d1<dimension_; d1++) {
                    gmeans[n*dimension_+d1] += (*((*this->trainingSet)(i)->second))(offset+t, d1);
                    for (int d2=0; d2<dimension_; d2++) {
                        components[n].covariance[d1*dimension_+d2] += (*((*this->trainingSet)(i)->second))(offset+t, d1) * (*((*this->trainingSet)(i)->second))(offset+t, d2);
                    }
                }
            }
            offset += step;
            factor[n] += step;
        }
    }
    
    for (int n=0; n<nbMixtureComponents_; n++)
        for (int d1=0; d1<dimension_; d1++) {
            gmeans[n*dimension_+d1] /= factor[n];
            for (int d2=0; d2<dimension_; d2++)
                components[n].covariance[d1*dimension_+d2] /= factor[n];
        }
    
    for (int n=0; n<nbMixtureComponents_; n++)
        for (int d1=0; d1<dimension_; d1++)
            for (int d2=0; d2<dimension_; d2++)
                components[n].covariance[d1*dimension_+d2] -= gmeans[n*dimension_+d1]*gmeans[n*dimension_+d2];
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
        components[c].covariance.assign(dimension_*dimension_, varianceOffset_absolute_/2.);
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
        for (mixture_iterator component = components.begin() ; component != components.end() ; ++component)
            component->updateInverseCovariance();
    } catch (exception& e) {
        throw runtime_error("Matrix inversion error: varianceoffset must be too small");
    }
}

#pragma mark > Performance
void GMM::regression(vector<float> const& observation_input, vector<float>& predicted_output)
{
    int dimension_output = dimension_ - dimension_input_;
    predicted_output.assign(dimension_output, 0.0);
    vector<float> tmp_predicted_output(dimension_output, 0.0);
    
    for (int c=0; c<nbMixtureComponents_; c++) {
        components[c].regression(observation_input, tmp_predicted_output);
        for (int d = 0; d < dimension_output; ++d)
        {
            predicted_output[d] += beta[c] * tmp_predicted_output[d];
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