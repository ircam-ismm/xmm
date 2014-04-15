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
                                           double offset)
: bimodal_(flags & BIMODAL),
  dimension_(dimension),
  dimension_input_(dimension_input),
  offset_(offset),
  covarianceDeterminant(0.),
  covarianceDeterminant_input_(0.)
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
    dst->offset_ = src.offset_;
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

double GaussianDistribution::offset() const
{
    return offset_;
}

void GaussianDistribution::set_offset(double offset)
{
    offset_ = offset;
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

double GaussianDistribution::likelihood_bimodal(const float* observation_input, const float* observation_output) const
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

void GaussianDistribution::regression(const float *observation_input, vector<float>& predicted_output) const
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
                tmp += inverseCovariance_input_[e * dimension_input_ + f]
                       * (observation_input[f] - mean[f]);
            }
            predicted_output[d] += covariance[(d + dimension_input_) * dimension_ + e]* tmp;
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
    json_gaussDist.push_back(JSONNode("offset", offset_));
    
    // Model Parameters
    json_gaussDist.push_back(vector2json(mean, "mean"));
    json_gaussDist.push_back(vector2json(covariance, "covariance"));
    
    return json_gaussDist;
}

 void GaussianDistribution::from_json(JSONNode root)
{
    try {
        assert(root.type() == JSON_NODE);
        JSONNode::iterator root_it = root.begin();
        
        // Get Dimension
        assert(root_it != root.end());
        assert(root_it->name() == "dimension");
        assert(root_it->type() == JSON_NUMBER);
        dimension_ = root_it->as_int();
        ++root_it;

        // Get Dimension of the input modality
        assert(root_it != root.end());
        assert(root_it->name() == "dimension_input");
        assert(root_it->type() == JSON_NUMBER);
        dimension_input_ = root_it->as_int();
        ++root_it;
        
        // Get Covariance Offset
        assert(root_it != root.end());
        assert(root_it->name() == "offset");
        assert(root_it->type() == JSON_NUMBER);
        offset_ = root_it->as_float();
        ++root_it;
        
        allocate();
        
        // Get Mean
        assert(root_it != root.end());
        assert(root_it->name() == "mean");
        assert(root_it->type() == JSON_ARRAY);
        json2vector(*root_it, mean, dimension_);
        ++root_it;
        
        // Get Covariance
        assert(root_it != root.end());
        assert(root_it->name() == "covariance");
        assert(root_it->type() == JSON_ARRAY);
        json2vector(*root_it, covariance, dimension_ * dimension_);
        
        updateInverseCovariance();
        
    } catch (JSONException &e) {
        throw JSONException(e);
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
}

void GaussianDistribution::setParametersToZero(bool initMeans)
{
    if (initMeans)
        fill(mean.begin(), mean.end(), 0.);
    fill(covariance.begin(), covariance.end(), 0.);
    fill(inverseCovariance_.begin(), inverseCovariance_.end(), 0.);
}

void GaussianDistribution::addOffset()
{
    for (int d = 0; d < dimension_; ++d)
    {
        covariance[d * dimension_ + d] += offset_;
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