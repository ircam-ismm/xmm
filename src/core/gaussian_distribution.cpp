/*
 * gaussian_distribution.cpp
 *
 * Multivariate Gaussian Distribution
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

#include "gaussian_distribution.h"
#include "matrix.h"
#include <algorithm>

#ifdef WIN32

#define M_PI 3.14159265358979323846264338328 /**< pi */
//#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#pragma mark Constructors
xmm::GaussianDistribution::GaussianDistribution(xmm_flags flags,
                                           unsigned int dimension,
                                           unsigned int dimension_input,
                                           double offset_relative,
                                           double offset_absolute,
                                           COVARIANCE_MODE covariance_mode)
: offset_relative(offset_relative),
  offset_absolute(offset_absolute),
  bimodal_(flags & BIMODAL),
  dimension_(dimension),
  dimension_input_(dimension_input),
  covarianceDeterminant(0.),
  covarianceDeterminant_input_(0.),
  covariance_mode_(covariance_mode)
{
    allocate();
}

xmm::GaussianDistribution::GaussianDistribution(GaussianDistribution const& src)
{
    _copy(this, src);
}

xmm::GaussianDistribution& xmm::GaussianDistribution::operator=(GaussianDistribution const& src)
{
    if(this != &src)
    {
        _copy(this, src);
    }
    return *this;
};

void xmm::GaussianDistribution::_copy(GaussianDistribution *dst, GaussianDistribution const& src)
{
    dst->dimension_ = src.dimension_;
    dst->offset_relative = src.offset_relative;
    dst->offset_absolute = src.offset_absolute;
    dst->bimodal_ = src.bimodal_;
    dst->dimension_input_ = src.dimension_input_;
    dst->scale = src.scale;
    dst->mean = src.mean;
    dst->covariance_mode_ = src.covariance_mode_;
    dst->covariance = src.covariance;
    dst->inverseCovariance_ = src.inverseCovariance_;
    dst->covarianceDeterminant = src.covarianceDeterminant;
    if (dst->bimodal_) {
        dst->covarianceDeterminant_input_ = src.covarianceDeterminant_input_;
        dst->inverseCovariance_input_ = src.inverseCovariance_input_;
        dst->output_variance = src.output_variance;
    }
}

xmm::GaussianDistribution::~GaussianDistribution()
{
}

#pragma mark Accessors
unsigned int xmm::GaussianDistribution::dimension() const
{
    return dimension_;
}

void xmm::GaussianDistribution::set_dimension(unsigned int dimension)
{
    dimension_ = dimension;
    allocate();
}

unsigned int xmm::GaussianDistribution::dimension_input() const
{
    return dimension_input_;
}

void xmm::GaussianDistribution::set_dimension_input(unsigned int dimension_input)
{
    if (dimension_input > dimension_ - 1)
        throw std::out_of_range("Input dimension is out of bounds.");
    dimension_input_ = dimension_input;
}

xmm::GaussianDistribution::COVARIANCE_MODE xmm::GaussianDistribution::get_covariance_mode() const
{
    return covariance_mode_;
}

void xmm::GaussianDistribution::set_covariance_mode(xmm::GaussianDistribution::COVARIANCE_MODE covariance_mode)
{
    if (covariance_mode == covariance_mode_) return;
    if (covariance_mode == DIAGONAL) {
        std::vector<double> new_covariance(dimension_);
        for (unsigned int d=0; d<dimension_; ++d) {
            new_covariance[d] = covariance[d*dimension_+d];
        }
        covariance = new_covariance;
        inverseCovariance_.resize(dimension_);
        if (bimodal_)
            inverseCovariance_input_.resize(dimension_input_);
    }
    if (covariance_mode == FULL) {
        std::vector<double> new_covariance(dimension_*dimension_, 0.0);
        for (unsigned int d=0; d<dimension_; ++d) {
            new_covariance[d*dimension_+d] = covariance[d];
        }
        covariance = new_covariance;
        inverseCovariance_.resize(dimension_ * dimension_);
        if (bimodal_)
            inverseCovariance_input_.resize(dimension_input_ * dimension_input_);
    }
    covariance_mode_ = covariance_mode;
    updateInverseCovariance();
    if (bimodal_) {
        updateOutputVariances();
    }
}

#pragma mark Likelihood & Regression
double xmm::GaussianDistribution::likelihood(const float* observation) const
{
    if (covarianceDeterminant == 0.0)
        throw std::runtime_error("Covariance Matrix is not invertible");
    
    double euclidianDistance(0.0);
    if (covariance_mode_ == FULL) {
        for (int l=0; l<dimension_; l++) {
            double tmp(0.0);
            for (int k=0; k<dimension_; k++) {
                tmp += inverseCovariance_[l*dimension_+k] * (observation[k] - mean[k]);
            }
            euclidianDistance += (observation[l] - mean[l]) * tmp;
        }
    } else {
        for (int l=0; l<dimension_; l++) {
            euclidianDistance += inverseCovariance_[l] * (observation[l] - mean[l]) * (observation[l] - mean[l]);
        }
    }
    
    double p = exp(-0.5 * euclidianDistance) / sqrt(covarianceDeterminant * pow(2*M_PI, double(dimension_)));
    
    if(p < 1e-180 || std::isnan(p) || std::isinf(fabs(p))) p = 1e-180;
    
    return p;
}

double xmm::GaussianDistribution::likelihood_input(const float* observation_input) const
{
    if (!bimodal_)
        throw std::runtime_error("'likelihood_input' can't be used when 'bimodal_' is off.");

    if (covarianceDeterminant_input_ == 0.0)
        throw std::runtime_error("Covariance Matrix of input modality is not invertible");
    
    double euclidianDistance(0.0);
    if (covariance_mode_ == FULL) {
        for (int l=0; l<dimension_input_; l++) {
            double tmp(0.0);
            for (int k=0; k<dimension_input_; k++) {
                tmp += inverseCovariance_input_[l*dimension_input_+k] * (observation_input[k] - mean[k]);
            }
            euclidianDistance += (observation_input[l] - mean[l]) * tmp;
        }
    } else {
        for (int l=0; l<dimension_input_; l++) {
            euclidianDistance += inverseCovariance_[l] * (observation_input[l] - mean[l]) * (observation_input[l] - mean[l]);
        }
    }
    
    double p = exp(-0.5 * euclidianDistance) / sqrt(covarianceDeterminant_input_ * pow(2*M_PI, double(dimension_input_)));
    
    if(p < 1e-180 || std::isnan(p) || std::isinf(fabs(p))) p = 1e-180;
    
    return p;
}

double xmm::GaussianDistribution::likelihood_bimodal(const float* observation_input,
                                                const float* observation_output) const
{
    if (!bimodal_)
        throw std::runtime_error("'likelihood_bimodal' can't be used when 'bimodal_' is off.");

    if (covarianceDeterminant == 0.0)
        throw std::runtime_error("Covariance Matrix is not invertible");
    
    int dimension_output = dimension_ - dimension_input_;
    double euclidianDistance(0.0);
    if (covariance_mode_ == FULL) {
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
    } else {
        for (int l=0; l<dimension_input_; l++) {
            euclidianDistance += inverseCovariance_[l] * (observation_input[l] - mean[l]) * (observation_input[l] - mean[l]);
        }
        for (int l=dimension_input_; l<dimension_; l++) {
            euclidianDistance += inverseCovariance_[l] * (observation_output[l-dimension_input_] - mean[l]) * (observation_output[l-dimension_input_] - mean[l]);
        }
    }
    
    
    double p = exp(-0.5 * euclidianDistance) / sqrt(covarianceDeterminant * pow(2*M_PI, (double)dimension_));
    
    if(p < 1e-180 || std::isnan(p) || std::isinf(fabs(p))) p = 1e-180;
    
    return p;
}

void xmm::GaussianDistribution::regression(std::vector<float> const& observation_input,
                                           std::vector<float>& predicted_output) const
{
    if (!bimodal_)
        throw std::runtime_error("'regression' can't be used when 'bimodal_' is off.");
    
    int dimension_output = dimension_ - dimension_input_;
    predicted_output.resize(dimension_output);
    
    if (covariance_mode_ == FULL) {
        for (int d=0; d<dimension_output; d++) {
            predicted_output[d] = mean[dimension_input_ + d];
            for (int e=0; e<dimension_input_; e++) {
                float tmp = 0.;
                for (int f=0; f<dimension_input_; f++) {
                    tmp += inverseCovariance_input_[e * dimension_input_ + f] * (observation_input[f] - mean[f]);
                }
                predicted_output[d] += covariance[(d + dimension_input_) * dimension_ + e] * tmp;
            }
        }
    } else {
        for (int d=0; d<dimension_output; d++) {
            predicted_output[d] = mean[dimension_input_ + d];
            predicted_output[d] += covariance[d + dimension_input_] * inverseCovariance_input_[d] * (observation_input[d] - mean[d]);
        }
    }
}

#pragma mark JSON I/O
JSONNode xmm::GaussianDistribution::to_json() const
{
    JSONNode json_gaussDist(JSON_NODE);
    json_gaussDist.set_name("GaussianDistribution");
    
    // Scalar Attributes
    json_gaussDist.push_back(JSONNode("dimension", dimension_));
    json_gaussDist.push_back(JSONNode("dimension_input", dimension_input_));
    json_gaussDist.push_back(JSONNode("offset_relative", offset_relative));
    json_gaussDist.push_back(JSONNode("offset_absolute", offset_absolute));
    json_gaussDist.push_back(vector2json(scale, "scale"));
    
    // Model Parameters
    json_gaussDist.push_back(vector2json(mean, "mean"));
    json_gaussDist.push_back(JSONNode("covariance_mode", covariance_mode_));
    json_gaussDist.push_back(vector2json(covariance, "covariance"));
    
    return json_gaussDist;
}

void xmm::GaussianDistribution::from_json(JSONNode root)
{
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong Node Type", root.name());
        JSONNode::iterator root_it = root.end();
        
        // Get Dimension
        root_it = root.find("dimension");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'dimension': was expecting 'JSON_NUMBER'", root_it->name());
        dimension_ = static_cast<unsigned int>(root_it->as_int());

        // Get Dimension of the input modality
        root_it = root.find("dimension_input");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'dimension_input': was expecting 'JSON_NUMBER'", root_it->name());
            dimension_input_ = static_cast<unsigned int>(root_it->as_int());
        }
        
        // Allocate Memory
        allocate();
        
        // Get Covariance Offset (Relative)
        root_it = root.find("offset_relative");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'offset_relative': was expecting 'JSON_NUMBER'", root_it->name());
        offset_relative = root_it->as_float();
        
        // Get Covariance Offset (Absolute)
        root_it = root.find("offset_absolute");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'offset_absolute': was expecting 'JSON_NUMBER'", root_it->name());
        offset_absolute = root_it->as_float();
        
        // Get Scale
        root_it = root.find("scale");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_ARRAY)
                throw JSONException("Wrong type for node 'scale': was expecting 'JSON_ARRAY'", root_it->name());
            json2vector(*root_it, scale, dimension_);
        }
        
        // Get Mean
        root_it = root.find("mean");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'mean': was expecting 'JSON_ARRAY'", root_it->name());
        json2vector(*root_it, mean, dimension_);
        
        // Get Covariance mode
        root_it = root.find("covariance_mode");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'covariance_mode': was expecting 'JSON_NUMBER'", root_it->name());
            covariance_mode_ = static_cast<xmm::GaussianDistribution::COVARIANCE_MODE>(root_it->as_int());
        } else {
            set_covariance_mode(FULL);
        }
        
        // Get Covariance
        root_it = root.find("covariance");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'covariance': was expecting 'JSON_ARRAY'", root_it->name());
        if (covariance_mode_ == FULL) {
            json2vector(*root_it, covariance, dimension_ * dimension_);
        } else {
            json2vector(*root_it, covariance, dimension_);
        }
        
        updateInverseCovariance();
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (std::exception &e) {
        throw JSONException(e, root.name());
    }
}

#pragma mark Utilities
void xmm::GaussianDistribution::allocate()
{
    mean.resize(dimension_);
    scale.assign(dimension_, 0.0);
    if (covariance_mode_ == FULL) {
        covariance.resize(dimension_ * dimension_);
        inverseCovariance_.resize(dimension_ * dimension_);
        if (bimodal_)
            inverseCovariance_input_.resize(dimension_input_ * dimension_input_);
    } else {
        covariance.resize(dimension_);
        inverseCovariance_.resize(dimension_);
        if (bimodal_)
            inverseCovariance_input_.resize(dimension_input_);
    }
}

void xmm::GaussianDistribution::addOffset()
{
    if (covariance_mode_ == FULL) {
        for (int d = 0; d < dimension_; ++d)
        {
            covariance[d * dimension_ + d] += std::max(offset_absolute, offset_relative * scale[d]);
        }
    }
    else
    {
        for (int d = 0; d < dimension_; ++d)
        {
            covariance[d] += std::max(offset_absolute, offset_relative * scale[d]);
        }
    }
}

void xmm::GaussianDistribution::updateInverseCovariance()
{
    if (covariance_mode_ == FULL)
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
    else // DIAGONAL COVARIANCE
    {
        covarianceDeterminant = 1.;
        covarianceDeterminant_input_ = 1.;
        for (unsigned int d=0; d<dimension_; ++d) {
            if (covariance[d] <= 0.0)
                throw std::runtime_error("Non-invertible matrix");
            inverseCovariance_[d] = 1. / covariance[d];
            covarianceDeterminant *= covariance[d];
            if (bimodal_ && d<dimension_input_) {
                inverseCovariance_input_[d] = 1. / covariance[d];
                covarianceDeterminant_input_ *= covariance[d];
            }
        }
    }
}

void xmm::GaussianDistribution::updateOutputVariances()
{
    if (!bimodal_)
        throw std::runtime_error("'updateOutputVariances' can't be used when 'bimodal_' is off.");
    
    unsigned int dimension_output = dimension_ - dimension_input_;
    
    // CASE: DIAGONAL COVARIANCE
    if (covariance_mode_ == DIAGONAL) {
        output_variance.resize(dimension_output);
        copy(covariance.begin()+dimension_input_, covariance.begin()+dimension_, output_variance.begin());
        return;
    }
    
    // CASE: FULL COVARIANCE
    Matrix<double> *inverseMat;
    double det;
    
    Matrix<double> cov_matrix_input(dimension_input_, dimension_input_, true);
    for (int d1=0; d1<dimension_input_; d1++) {
        for (int d2=0; d2<dimension_input_; d2++) {
            cov_matrix_input._data[d1*dimension_input_+d2] = covariance[d1 * dimension_ + d2];
        }
    }
    inverseMat = cov_matrix_input.pinv(&det);
    Matrix<double> covariance_gs(dimension_input_, dimension_output, true);
    for (int d1=0; d1<dimension_input_; d1++) {
        for (int d2=0; d2<dimension_output; d2++) {
            covariance_gs._data[d1*dimension_output+d2] = covariance[d1 * dimension_ + dimension_input_ + d2];
        }
    }
    Matrix<double> covariance_sg(dimension_output, dimension_input_, true);
    for (int d1=0; d1<dimension_output; d1++) {
        for (int d2=0; d2<dimension_input_; d2++) {
            covariance_gs._data[d1*dimension_input_+d2] = covariance[(dimension_input_ + d1) * dimension_ + d2];
        }
    }
    Matrix<double> *tmptmptmp = inverseMat->product(&covariance_gs);
    Matrix<double> *covariance_mod = covariance_sg.product(tmptmptmp);
    output_variance.resize(dimension_output);
    for (int d=0; d<dimension_output; d++) {
        output_variance[d] = covariance[(dimension_input_ + d) * dimension_ + dimension_input_ + d] - covariance_mod->data[d * dimension_output + d];
    }
    delete inverseMat;
    delete covariance_mod;
    delete tmptmptmp;
    inverseMat = NULL;
    covariance_mod = NULL;
    tmptmptmp = NULL;
}

xmm::Ellipse xmm::GaussianDistribution::ellipse(unsigned int dimension1,
                                      unsigned int dimension2)
{
    if (dimension1 >= dimension_ || dimension2 >= dimension_)
        throw std::out_of_range("dimensions out of range");
    
    Ellipse gaussian_ellipse_95;
    gaussian_ellipse_95.x = mean[dimension1];
    gaussian_ellipse_95.y = mean[dimension2];
    
    // Represent 2D covariance with square matrix
    // |a b|
    // |b c|
    double a, b, c;
    if (covariance_mode_ == FULL) {
        a = covariance[dimension1 * dimension_ + dimension1];
        b = covariance[dimension1 * dimension_ + dimension2];
        c = covariance[dimension2 * dimension_ + dimension2];
    } else {
        a = covariance[dimension1];
        b = 0.0;
        c = covariance[dimension2];
    }
    // Compute Eigen Values to get width, height and angle
    double trace = a+c;
    double determinant = a*c - b*b;
    double eigenVal1 = 0.5 * (trace + sqrt(trace*trace - 4*determinant));
    double eigenVal2 = 0.5 * (trace - sqrt(trace*trace - 4*determinant));
    gaussian_ellipse_95.width = 2 * sqrt(5.991 * eigenVal1);
    gaussian_ellipse_95.height = 2 * sqrt(5.991 * eigenVal2);
    gaussian_ellipse_95.angle = atan(b / (eigenVal1 - c));
    
    return gaussian_ellipse_95;
}

void xmm::GaussianDistribution::make_bimodal(unsigned int dimension_input)
{
    if (bimodal_)
        throw std::runtime_error("The model is already bimodal");
    if (dimension_input >= dimension_)
        throw std::out_of_range("Request input dimension exceeds the current dimension");
    this->bimodal_ = true;
    this->dimension_input_ = dimension_input;
    if (covariance_mode_ == FULL) {
        this->inverseCovariance_input_.resize(dimension_input*dimension_input);
    } else {
        this->inverseCovariance_input_.resize(dimension_input);
    }
    this->updateInverseCovariance();
    this->updateOutputVariances();
}

void xmm::GaussianDistribution::make_unimodal()
{
    if (!bimodal_)
        throw std::runtime_error("The model is already unimodal");
    this->bimodal_ = false;
    this->dimension_input_ = 0;
    this->inverseCovariance_input_.clear();
}

xmm::GaussianDistribution xmm::GaussianDistribution::extract_submodel(std::vector<unsigned int>& columns) const
{
    if (columns.size() > dimension_)
        throw std::out_of_range("requested number of columns exceeds the dimension of the current model");
    for (unsigned int column=0; column<columns.size(); ++column) {
        if (columns[column] >= dimension_)
            throw std::out_of_range("Some column indices exceeds the dimension of the current model");
    }
    size_t new_dim =columns.size();
    GaussianDistribution target_distribution(NONE, static_cast<unsigned int>(new_dim), 0, offset_relative, offset_absolute);
    target_distribution.allocate();
    for (unsigned int new_index1=0; new_index1<new_dim; ++new_index1) {
        unsigned int col_index1 = columns[new_index1];
        target_distribution.mean[new_index1] = mean[col_index1];
        target_distribution.scale[new_index1] = scale[col_index1];
        if (covariance_mode_ == FULL) {
            for (unsigned int new_index2=0; new_index2<new_dim; ++new_index2) {
                unsigned int col_index2 = columns[new_index2];
                target_distribution.covariance[new_index1*new_dim+new_index2] = covariance[col_index1*dimension_+col_index2];
            }
        } else {
            target_distribution.covariance[new_index1] = covariance[col_index1];
        }
    }
    try {
        target_distribution.updateInverseCovariance();
    } catch (std::exception const& e) {
    }
    return target_distribution;
}

xmm::GaussianDistribution xmm::GaussianDistribution::extract_submodel_input() const
{
    if (!bimodal_)
        throw std::runtime_error("The distribution needs to be bimodal");
    std::vector<unsigned int> columns_input(dimension_input_);
    for (unsigned int i=0; i<dimension_input_; ++i) {
        columns_input[i] = i;
    }
    return extract_submodel(columns_input);
}

xmm::GaussianDistribution xmm::GaussianDistribution::extract_submodel_output() const
{
    if (!bimodal_)
        throw std::runtime_error("The distribution needs to be bimodal");
    std::vector<unsigned int> columns_output(dimension_ - dimension_input_);
    for (unsigned int i=dimension_input_; i<dimension_; ++i) {
        columns_output[i-dimension_input_] = i;
    }
    return extract_submodel(columns_output);
}

xmm::GaussianDistribution xmm::GaussianDistribution::extract_inverse_model() const
{
    if (!bimodal_)
        throw std::runtime_error("The distribution needs to be bimodal");
    std::vector<unsigned int> columns(dimension_);
    for (unsigned int i=0; i<dimension_-dimension_input_; ++i) {
        columns[i] = i+dimension_input_;
    }
    for (unsigned int i=dimension_-dimension_input_, j=0; i<dimension_; ++i, ++j) {
        columns[i] = j;
    }
    GaussianDistribution target_distribution = extract_submodel(columns);
    target_distribution.make_bimodal(dimension_-dimension_input_);
    return target_distribution;
}

