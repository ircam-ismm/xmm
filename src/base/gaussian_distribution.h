//
// gaussian_distribution.h
//
// Multivariate Gaussian Distribution
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
//

#ifndef __mhmm__gaussian_distribution__
#define __mhmm__gaussian_distribution__

#include "json_utilities.h"
#include "mbd_common.h"

/**
 * default offset for covariance matrix
 */
const double GAUSSIAN_DEFAULT_VARIANCE_OFFSET_RELATIVE = 1.;
const double GAUSSIAN_DEFAULT_VARIANCE_OFFSET_ABSOLUTE = 1.e-3;

/**
 * Should avoid probabilities > 1 in most cases
 * @todo Weird stuff, to check.
 */
const double EPSILON_GAUSSIAN = 1.0e-20;

/**
 * @ingroup ModelBase
 * @struct Ellipse
 * @brief Simple structure for storing Ellipse parameters
 */
struct Ellipse {
    /**
     * @brief x center position
     */
    float x;
    
    /**
     * @brief y center position
     */
    float y;
    
    /**
     * @brief width: minor axis length
     */
    float width;
    
    /**
     * @brief height: major axis length
     */
    float height;
    
    /**
     * @brief angle (radians)
     */
    float angle;
};

/**
 * @ingroup ModelBase
 * @class GaussianDistribution
 * @brief Multivariate Gaussian Distribution
 * @details Full covariance, optionally multimodal with support for regression
 */
class GaussianDistribution : public Writable
{
public:
#pragma mark > Constructors
    /*@{*/
    /** @name Constructors */
    /**
     * @brief Default Constructor
     * @param flags construction flags. Use the flag 'BIMODAL' for use with regression.
     * @param dimension dimension of the distribution
     * @param offset_relative Offset added to diagonal covariance (proportional to variance)
     * @param offset_absolute Offset added to diagonal covariance (minimum value)
     * @param dimension_input dimension of the input modality in bimodal mode.
     */
    GaussianDistribution(rtml_flags flags = NONE,
                         unsigned int dimension=1,
                         unsigned int dimension_input = 0,
                         double offset_relative = GAUSSIAN_DEFAULT_VARIANCE_OFFSET_RELATIVE,
                         double offset_absolute = GAUSSIAN_DEFAULT_VARIANCE_OFFSET_ABSOLUTE);
    
    /**
     * @brief Copy constructor
     * @param src source distribution
     */
    GaussianDistribution(GaussianDistribution const& src);
    
    /**
     * @brief Assignment
     * @param src source distribution
     */
    GaussianDistribution& operator=(GaussianDistribution const& src);
    
    /**
     * @brief Copy between 2 Gaussian Distributions
     * @param dst destination distribution
     * @param src source distribution
     */
    void _copy(GaussianDistribution *dst, GaussianDistribution const& src);
    
    /**
     * @brief Destructor
     */
    ~GaussianDistribution();
    
    /*@}*/
    
#pragma mark > Accessors
    /*@{*/
    /** @name Accessors */
    /**
     * @brief Get Dimension of the distribution
     * @return dimension
     */
    unsigned int dimension() const;
    
    /**
     * @brief Set Dimension of the distribution
     */
    void set_dimension(unsigned int dimension);
    
    /**
     * @brief Get Dimension of the input modality
     * @return dimension
     */
    unsigned int dimension_input() const;
    
    /**
     * @brief Set Dimension of the input modality
     * @param dimension_input dimension of the input modality
     * @throws out_of_range if the dimension is superior to the total dimension
     */
    void set_dimension_input(unsigned int dimension_input);
    
    /*@}*/

#pragma mark > Likelihood & Regression
    /*@{*/
    /** @name Likelihood & Regression */
    /**
     * @brief Get Likelihood of a data vector
     * @param observation data observation (must be of size @a dimension)
     * @return likelihood
     * @throws runtime_error if the Covariance Matrix is not invertible
     */
    double likelihood(const float* observation) const;
    
    /**
     * @brief Get Likelihood of a data vector for input modality
     * @param observation_input observation (must be of size @a dimension_input)
     * @return likelihood
     * @throws runtime_error if the Covariance Matrix of the input modality is not invertible
     * @throws runtime_error if the model is not bimodal
     */
    double likelihood_input(const float* observation_input) const;
    
    /**
     * @brief Get Likelihood of a data vector for bimodal mode
     * @param observation_input observation of the input modality
     * @param observation_output observation of the output modality
     * @throws runtime_error if the Covariance Matrix is not invertible
     * @throws runtime_error if the model is not bimodal
     * @return likelihood
     */
    double likelihood_bimodal(const float* observation_input, const float* observation_output) const;
    
    /**
     * @brief Linear Regression using the Gaussian Distribution (covariance-based)
     * @param observation_input input observation (must be of size: @a dimension_input)
     * @param predicted_output predicted output vector (size: dimension-dimension_input)
     * @throws runtime_error if the model is not bimodal
     */
    void regression(vector<float> const& observation_input, vector<float>& predicted_output) const;
    
    /*@}*/

#pragma mark > JSON I/O
    /*@{*/
    /** @name JSON I/O */
    /**
     * @brief Write to JSON Node
     * @return The JSON Node containing the Gaussian Distribution parameters
     */
    JSONNode to_json() const;
    
    /**
     * @brief Write to JSON Node
     * @details allocate model parameters and updates inverse Covariances
     * @param root JSON Node containing model information and parameters
     * @throws JSONException if the JSONNode has a wrong format
     */
    void from_json(JSONNode root);
    
    /*@}*/

#pragma mark > Utilities
    /*@{*/
    /** @name Utilities */
    /**
     * @brief Resize Mean and Covariance Vectors to appropriate dimension.
     */
    void allocate();
    
    /**
     * @brief Add @a offset to the diagonal of the covariance matrix
     * @details Ensures convergence + generalization on few examples
     */
    void addOffset();
    
    /**
     * @brief Compute inverse covariance matrix
     * @throws runtime_error if the covariance matrix is not invertible
     */
    void updateInverseCovariance();
    
    /**
     * @brief Compute the 95% Confidence Interval ellipse of the Gaussian
     * @details the ellipse is 2D, and is therefore projected over 2 axes
     * @param dimension1 index of the first axis
     * @param dimension2 index of the second axis
     * @throws out_of_range if the dimensions are out of bounds
     * @return ellipse parameters
     */
    Ellipse ellipse(unsigned int dimension1,
                    unsigned int dimension2);

    /*@}*/

#pragma mark -
#pragma mark === Public Attributes ===
    /** @name Public Attributes */
    
    /**
     * @brief Mean of the Gaussian Distribution
     */
    vector<double> mean;
    
    /**
     * @brief Covariance Matrix of the Gaussian Distribution
     */
    vector<double> covariance;
    
    /**
     * @brief Offset added to diagonal covariance (proportional to variance)
     */
    double offset_relative;
    
    /**
     * @brief Offset added to diagonal covariance (minimum value)
     */
    double offset_absolute;
    
    /**
     * @brief Scaling of each dimension of the Gaussian Distribution (used of variance offsets)
     */
    vector<float> scale;
    
    /**
     * @brief specifies the weight of the regression part in output modality estimation.
     * A zero weight correspond to estimation using the means only.
     */
    double weight_regression;
    
#pragma mark -
#pragma mark === Private Attributes ===
private:
    /**
     * @brief Defines if regression parameters need to be computed
     */
    bool bimodal_;
    
    /**
     * @brief Dimension of the distribution
     */
    unsigned int dimension_;
    
    /**
     * @brief Dimension of the input modality
     */
    unsigned int dimension_input_;
    
    /**
     * @brief Determinant of the covariance matrix
     */
    double covarianceDeterminant;
    
    /**
     * @brief Inverse covariance matrix
     */
    vector<double> inverseCovariance_;
    
    /**
     * @brief Determinant of the covariance matrix of the input modality
     */
    double covarianceDeterminant_input_;
    
    /**
     * @brief Inverse covariance matrix of the input modality
     */
    vector<double> inverseCovariance_input_;
};

#endif
