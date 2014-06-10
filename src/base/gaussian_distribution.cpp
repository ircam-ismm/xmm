//
// gaussian_distribution.h
//
// Multivariate Gaussian Distribution
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#include "gaussian_distribution.h"
#include "matrix.h"

#pragma mark Constructors
GaussianDistribution::GaussianDistribution(rtml_flags flags,
                                           unsigned int dimension,
                                           unsigned int dimension_input,
                                           double offset_relative,
                                           double offset_absolute)
: bimodal_(flags & BIMODAL),
  dimension_(dimension),
  offset_relative(offset_relative),
  offset_absolute(offset_absolute),
  dimension_input_(dimension_input),
  covarianceDeterminant(0.),
  covarianceDeterminant_input_(0.),
  weight_regression(1.)
{
    allocate();
}

GaussianDistribution::GaussianDistribution(GaussianDistribution const& src)
{
    _copy(this, src);
}

GaussianDistribution& GaussianDistribution::operator=(GaussianDistribution const& src)
{
    if(this != &src)
    {
        _copy(this, src);
    }
    return *this;
};

void GaussianDistribution::_copy(GaussianDistribution *dst, GaussianDistribution const& src)
{
    dst->dimension_ = src.dimension_;
    dst->offset_relative = src.offset_relative;
    dst->offset_absolute = src.offset_absolute;
    dst->weight_regression = src.weight_regression;
    dst->bimodal_ = src.bimodal_;
    dst->dimension_input_ = src.dimension_input_;
    dst->mean = src.mean;
    dst->covariance = src.covariance;
    dst->inverseCovariance_ = src.inverseCovariance_;
    dst->covarianceDeterminant = src.covarianceDeterminant;
    if (dst->bimodal_) {
        dst->covarianceDeterminant_input_ = src.covarianceDeterminant_input_;
        dst->inverseCovariance_input_ = src.inverseCovariance_input_;
    }
    
    dst->allocate();
}

GaussianDistribution::~GaussianDistribution()
{
    mean.clear();
    covariance.clear();
    inverseCovariance_.clear();
    if (bimodal_)
        inverseCovariance_input_.clear();
}

#pragma mark Accessors
unsigned int GaussianDistribution::dimension() const
{
    return dimension_;
}

void GaussianDistribution::set_dimension(unsigned int dimension)
{
    dimension_ = dimension;
    allocate();
}

unsigned int GaussianDistribution::dimension_input() const
{
    return dimension_input_;
}

void GaussianDistribution::set_dimension_input(unsigned int dimension_input)
{
    if (dimension_input > dimension_ - 1)
        throw out_of_range("Input dimension is out of bounds.");
    dimension_input_ = dimension_input;
}

#pragma mark Likelihood & Regression
double GaussianDistribution::likelihood(const float* observation) const
{
    if (covarianceDeterminant == 0.0)
        throw runtime_error("Covariance Matrix is not invertible");
    
    double euclidianDistance(0.0);
    for (int l=0; l<dimension_; l++) {
        double tmp(0.0);
        for (int k=0; k<dimension_; k++) {
            tmp += inverseCovariance_[l*dimension_+k] * (observation[k] - mean[k]);
        }
        euclidianDistance += (observation[l] - mean[l]) * tmp;
    }
    
    double p = exp(-0.5 * euclidianDistance) * EPSILON_GAUSSIAN / sqrt(covarianceDeterminant * pow(2*M_PI, double(dimension_)));
    
    if(p < 1e-80 || isnan(p) || isinf(abs(p))) p = 1e-80;
    
    return p;
}

double GaussianDistribution::likelihood_input(const float* observation_input) const
{
    if (!bimodal_)
        throw runtime_error("'likelihood_input' can't be used when 'useRegression' is off.");

    if (covarianceDeterminant_input_ == 0.0)
        throw runtime_error("Covariance Matrix of input modality is not invertible");
    
    double euclidianDistance(0.0);
    for (int l=0; l<dimension_input_; l++) {
        double tmp(0.0);
        for (int k=0; k<dimension_input_; k++) {
            tmp += inverseCovariance_input_[l*dimension_input_+k] * (observation_input[k] - mean[k]);
        }
        euclidianDistance += (observation_input[l] - mean[l]) * tmp;
    }
    
    double p = exp(-0.5 * euclidianDistance) * EPSILON_GAUSSIAN / sqrt(covarianceDeterminant_input_ * pow(2*M_PI, double(dimension_input_)));
    
    if(p < 1e-80 || isnan(p) || isinf(abs(p))) p = 1e-80;
    
    return p;
}

double GaussianDistribution::likelihood_bimodal(const float* observation_input,
                                                const float* observation_output) const
{
    if (!bimodal_)
        throw runtime_error("'likelihood_bimodal' can't be used when 'useRegression' is off.");

    if (covarianceDeterminant == 0.0)
        throw runtime_error("Covariance Matrix is not invertible");
    
    int dimension_output = dimension_ - dimension_input_;
    double euclidianDistance(0.0);
    for (int l=0; l<dimension_; l++) {
        double tmp(0.0);
        for (int k=0; k<dimension_input_; k++) {
            tmp += inverseCovariance_[l*dimension_+k] * (observation_input[k] - mean[k]);
        }
        for (int k=0; k<dimension_output; k++) {
            tmp += inverseCovariance_[l * dimension_ + dimension_input_ + k] * (observation_output[k] - mean[dimension_input_ + k]);
        }
        if (l<dimension_input_)
            euclidianDistance += (observation_input[l] - mean[l]) * tmp;
        else
            euclidianDistance += (observation_output[l-dimension_input_] - mean[l]) * tmp;
    }
    
    double p = exp(-0.5 * euclidianDistance) * EPSILON_GAUSSIAN / sqrt(covarianceDeterminant * pow(2*M_PI, (double)dimension_));
    
    return p;
}

void GaussianDistribution::regression(vector<float> const& observation_input, vector<float>& predicted_output) const
{
    if (!bimodal_)
        throw runtime_error("'regression' can't be used when 'useRegression' is off.");

    int dimension_output = dimension_ - dimension_input_;
    predicted_output.resize(dimension_output);

    for (int d=0; d<dimension_output; d++) {
        predicted_output[d] = mean[dimension_input_ + d];
        for (int e=0; e<dimension_input_; e++) {
            float tmp = 0.;
            for (int f=0; f<dimension_input_; f++) {
                if (e == f && covariance[e * dimension_ + e] > offset_absolute) {
                    tmp += inverseCovariance_input_[e * dimension_input_ + e] * (covariance[e * dimension_ + e] / (covariance[e * dimension_ + e] - max(scale[e] * offset_relative, offset_absolute))) * (observation_input[f] - mean[f]);
                } else
                    tmp += inverseCovariance_input_[e * dimension_input_ + f] * (observation_input[f] - mean[f]);
            }
            predicted_output[d] += weight_regression * covariance[(d + dimension_input_) * dimension_ + e] * tmp;
        }
    }
}

#pragma mark JSON I/O
JSONNode GaussianDistribution::to_json() const
{
    JSONNode json_gaussDist(JSON_NODE);
    json_gaussDist.set_name("GaussianDistribution");
    
    // Scalar Attributes
    json_gaussDist.push_back(JSONNode("dimension", dimension_));
    json_gaussDist.push_back(JSONNode("dimension_input", dimension_input_));
    json_gaussDist.push_back(JSONNode("offset_relative", offset_relative));
    json_gaussDist.push_back(JSONNode("offset_absolute", offset_absolute));
    
    // Model Parameters
    json_gaussDist.push_back(vector2json(mean, "mean"));
    json_gaussDist.push_back(vector2json(covariance, "covariance"));
    
    return json_gaussDist;
}

 void GaussianDistribution::from_json(JSONNode root)
{
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong Node Type", root.name());
        JSONNode::iterator root_it = root.begin();
        
        // Get Dimension
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "dimension")
            throw JSONException("Wrong name: was expecting 'dimension'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        dimension_ = root_it->as_int();
        ++root_it;

        // Get Dimension of the input modality
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "dimension_input")
            throw JSONException("Wrong name: was expecting 'dimension_input'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        dimension_input_ = root_it->as_int();
        ++root_it;
        
        // Get Covariance Offset (Relative)
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "offset_relative")
            throw JSONException("Wrong name: was expecting 'offset_relative'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        offset_relative = root_it->as_float();
        ++root_it;
        
        // Get Covariance Offset (Absolute)
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "offset_absolute")
            throw JSONException("Wrong name: was expecting 'offset_absolute'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        offset_absolute = root_it->as_float();
        ++root_it;
        
        allocate();
        
        // Get Mean
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "mean")
            throw JSONException("Wrong name: was expecting 'mean'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        json2vector(*root_it, mean, dimension_);
        ++root_it;
        
        // Get Covariance
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "covariance")
            throw JSONException("Wrong name: was expecting 'covariance'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        json2vector(*root_it, covariance, dimension_ * dimension_);
        
        updateInverseCovariance();
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
}

#pragma mark Utilities
void GaussianDistribution::allocate()
{
    mean.resize(dimension_);
    covariance.resize(dimension_ * dimension_);
    inverseCovariance_.resize(dimension_ * dimension_);
    if (bimodal_)
        inverseCovariance_input_.resize(dimension_input_ * dimension_input_);
    scale.assign(dimension_, 1.0);
}

void GaussianDistribution::addOffset()
{
    for (int d = 0; d < dimension_; ++d)
    {
        covariance[d * dimension_ + d] += offset_relative * scale[d];
        if (covariance[d * dimension_ + d] < offset_absolute)
            covariance[d * dimension_ + d] = offset_absolute;
    }
}

void GaussianDistribution::updateInverseCovariance()
{
    Matrix<double> cov_matrix(dimension_, dimension_, false);
    
    Matrix<double> *inverseMat;
    double det;
    
    cov_matrix.data = covariance.begin();
    inverseMat = cov_matrix.pinv(&det);
    covarianceDeterminant = det;
    copy(inverseMat->data,
         inverseMat->data + dimension_ * dimension_,
         inverseCovariance_.begin());
    delete inverseMat;
    inverseMat = NULL;
    
    // If regression active: create inverse covariance matrix for input modality.
    if (bimodal_)
    {
        Matrix<double> cov_matrix_input(dimension_input_, dimension_input_, true);
        for (int d1=0; d1<dimension_input_; d1++) {
            for (int d2=0; d2<dimension_input_; d2++) {
                cov_matrix_input._data[d1*dimension_input_+d2] = covariance[d1 * dimension_ + d2];
            }
        }
        inverseMat = cov_matrix_input.pinv(&det);
        covarianceDeterminant_input_ = det;
        copy(inverseMat->data,
             inverseMat->data + dimension_input_ * dimension_input_,
             inverseCovariance_input_.begin());
        delete inverseMat;
        inverseMat = NULL;
    }
}