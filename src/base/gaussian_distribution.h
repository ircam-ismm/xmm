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
const double GAUSSIAN_DEFAULT_COVARIANCE_OFFSET = 0.01;

/**
 * Should avoid probabilities > 1 in most cases
 * @todo Weird stuff, to check.
 */
const double EPSILON_GAUSSIAN = 1.0e-40;

/**
 * @ingroup ModelBase
 * @class GaussianDistribution
 * @brief Multivariate Gaussian Distribution
 * @details Full covariance, optionally multimodal with support for regression
 */
class GaussianDistribution
{
public:
#pragma mark > Constructors
    /*@{*/
    /** @name Constructors */
    /**
     * @brief Default Constructor
     * @param flags construction flags. Use the flag 'BIMODAL' for use with regression.
     * @param dimension dimension of the distribution
     * @param offset offset added to the covariances to ensure convergence / avoid numeric errors
     * @param dimension_input dimension of the input modality in bimodal mode.
     * @todo change useRegression boolean to BIMODAL flag
     */
    GaussianDistribution(rtml_flags flags = NONE,
                         unsigned int dimension=1,
                         unsigned int dimension_input = 0,
                         double offset = GAUSSIAN_DEFAULT_COVARIANCE_OFFSET);
    
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
     * @todo  check the 'EPSILON_GAUSSIAN' stuff
     */
    double likelihood(const float* observation) const;
    
    /**
     * @brief Get Likelihood of a data vector for input modality
     * @param observation_input observation (must be of size @a dimension_input)
     * @return likelihood
     * @throws runtime_error if the Covariance Matrix of the input modality is not invertible
     * @throws runtime_error if the model is not bimodal
     * @todo  check the 'EPSILON_GAUSSIAN' stuff
     */
    double likelihood_input(const float* observation_input) const;
    
    /**
     * @brief Get Likelihood of a data vector for bimodal mode
     * @param observation_input observation of the input modality
     * @param observation_output observation of the output modality
     * @throws runtime_error if the Covariance Matrix is not invertible
     * @throws runtime_error if the model is not bimodal
     * @todo  check the 'EPSILON_GAUSSIAN' stuff
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
     * @return The JSON Node containing the Gaussian Distribution parameters
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
     * @brief Offset added to diagonal covariance
     */
    double offset;
    
    vector<float> scale;
    
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
