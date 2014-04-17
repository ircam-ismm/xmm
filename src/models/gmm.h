//
// gmm.h
//
// Gaussian Mixture Model
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
//

#ifndef mhmm_gmm_h
#define mhmm_gmm_h

#include "gaussian_distribution.h"
#include "probabilistic_model.h"

const int GMM_DEFAULT_NB_MIXTURE_COMPONENTS = 1;

/**
 * @defgroup GMM Gaussian Mixture Model
 */

/**
 * @ingroup GMM
 * @class GMM
 * @brief Gaussian Mixture Model
 * @details Multivariate Gaussian Mixture Model. Supports Bimodal data and Gaussian Mixture Regression.
 * Can be either autonomous or a state of a HMM: defines observation probabilities for each state.
 */
class GMM : public ProbabilisticModel {
public:
    friend class HMM;
    friend class HierarchicalHMM;
    
    /**
     * @brief Iterator over the phrases of the training set.
     */
    typedef map<int, Phrase* >::iterator phrase_iterator;
    
    /**
     * @brief Iterator over Gaussian Mixture Components
     */
    typedef vector<GaussianDistribution>::iterator mixture_iterator;
    
#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Constructors
    /*@{*/
    /** @name Constructors */
    /**
     * @brief Constructor
     * @param flags Construction Flags: use 'BIMODAL' for use with Gaussian Mixture Regression.
     * @param trainingSet training set associated with the model
     * @param nbMixtureComponents number of mixture components
     * @param covarianceOffset offset added to the diagonal of covariances matrices (useful to guarantee convergence)
     */
    GMM(rtml_flags flags = NONE,
        TrainingSet *trainingSet=NULL,
        int nbMixtureComponents = GMM_DEFAULT_NB_MIXTURE_COMPONENTS,
        float covarianceOffset = GAUSSIAN_DEFAULT_COVARIANCE_OFFSET);
    
    /**
     * @brief Copy constructor
     * @param src Source GMM
     */
    GMM(GMM const& src);
    
    /**
     * @brief Assignment
     * @param src Source GMM
     */
    GMM& operator=(GMM const& src);
    
    /**
     * @brief Destructor
     */
    virtual ~GMM();
    
    /*@}*/

#pragma mark > Accessors
    /*@{*/
    /** @name Accessors */
    /**
     * @brief Get the number of Gaussian mixture Components
     * @return number of Gaussian mixture components
     */
    int get_nbMixtureComponents() const;
    
    /**
     * @brief Get Offset added to covariance matrices for convergence
     * @return Offset added to covariance matrices for convergence
     */
    float get_covarianceOffset() const;
    
    /**
     * @brief Set the number of mixture components of the model
     * @warning sets the model to be untrained.
     * @todo : change this untrained behavior via mirror models for training?
     * @param nbMixtureComponents number of Gaussian Mixture Components
     * @throws invalid_argument if nbMixtureComponents is <= 0
     */
    void set_nbMixtureComponents(int nbMixtureComponents);
    
    /**
     * @brief Set the offset to add to the covariance matrices
     * @param covarianceOffset offset to add to the diagonal of covariance matrices
     * @throws invalid_argument if the covariance offset is <= 0
     */
    void set_covarianceOffset(float covarianceOffset);
    
    /*@}*/

#pragma mark > Performance
    /*@{*/
    /** @name Performance */
    /**
     * @brief Initialize performance mode
     */
    void performance_init();
    
    /**
     * @brief Main Play function: performs recognition (unimodal mode) or regression (bimodal mode)
     * @details The predicted output is stored in the observation vector in bimodal mode
     * @param observation observation (must allocated to size 'dimension')
     * @return instantaneous likelihood
     */
    double performance_update(vector<float> const& observation);

    /*@}*/

#pragma mark > JSON I/O
    /*@{*/
    /** @name JSON I/O */
    /**
     * @brief Write to JSON Node
     * @return JSON Node containing model information and parameters
     */
    JSONNode to_json() const;
    
    /**
     * @brief Read from JSON Node
     * @details allocate model parameters and updates inverse Covariances
     * @param root JSON Node containing model information and parameters
     * @throws JSONException if the JSONNode has a wrong format
     */
    virtual void from_json(JSONNode root);
    
    /*@}*/
    
#pragma mark -
#pragma mark === Public attributes ===
    /**
     * @brief Vector of Gaussian Mixture Components
     */
    vector<GaussianDistribution> components;
    
    /**
     * @brief Mixture Coefficients
     */
    vector<float> mixtureCoeffs;
    
    /**
     * @brief Beta probabilities: estimated likelihoods of each component
     */
    vector<double> beta;
    
protected:
#pragma mark -
#pragma mark === Protected Methods ===
#pragma mark > Utilities
    /*@{*/
    /** @name Copy between models */
    /**
     * @brief Copy between 2 GMMs
     * @param dst Destination GMM
     * @param src Source GMM
     */
    using ProbabilisticModel::_copy;
    virtual void _copy(GMM *dst, GMM const& src);
    
    /*@}*/
    
    /*@{*/
    /** @name Utilities */
    /**
     @brief Allocate model parameters
     */
    void allocate();
    
    /**
     * @brief Observation probability
     * @param observation observation vector (must be of size 'dimension')
     * @param mixtureComponent index of the mixture component. if unspecified or negative,
     * full mixture observation probability is computed
     * @return likelihood of the observation given the model
     * @throws out_of_range if the index of the Gaussian Mixture Component is out of bounds
     * @throws runtime_error if the Covariance Matrix is not invertible
     */
    double obsProb(const float* observation, int mixtureComponent=-1);
    
    /**
     * @brief Observation probability on the input modality
     * @param observation_input observation vector of the input modality (must be of size 'dimension_input')
     * @param mixtureComponent index of the mixture component. if unspecified or negative,
     * full mixture observation probability is computed
     * @return likelihood of the observation of the input modality given the model
     * @throws runtime_error if the model is not bimodal
     * @throws runtime_error if the Covariance Matrix of the input modality is not invertible
     */
    double obsProb_input(const float*_input, int mixtureComponent=-1);
    
    /**
     * @brief Observation probability for bimodal mode
     * @param observation_input observation vector of the input modality (must be of size 'dimension_input')
     * @param observation_output observation vector of the input output (must be of size 'dimension - dimension_input')
     * @param mixtureComponent index of the mixture component. if unspecified or negative,
     * full mixture observation probability is computed
     * @return likelihood of the observation of the input modality given the model
     * @throws runtime_error if the model is not bimodal
     * @throws runtime_error if the Covariance Matrix is not invertible
     */
    double obsProb_bimodal(const float*_input, const float*_output, int mixtureComponent=-1);
    
    /*@}*/

#pragma mark > Training
    /*@{*/
    /** @name Training: internal methods */
    /**
     * @brief Initialize the means of the Gaussian components with the first phrase of the training set
     */
    void initMeansWithFirstPhrase();
    
    /**
     * @brief Set model parameters to zero, except means (sometimes means are not re-estimated in the training algorithm)
     * @todo Add initMeans as a parameter? >> look in GMM training where it is done.
     */
    void setParametersToZero();
    
    /**
     * @brief Initialize the EM Training Algorithm
     * @details Initializes the Gaussian Components from the first phrase
     * of the Training Set
     */
    virtual void train_EM_init();
    
    /**
     * @brief Update Function of the EM algorithm
     * @return likelihood of the data given the current parameters (E-step)
     */
    double train_EM_update();
    
    /**
     * @brief Initialize model parameters to default values.
     * @details Mixture coefficients are then equiprobable
     */
    virtual void initParametersToDefault();
    
    /**
     * @brief Normalize mixture coefficients
     */
    void normalizeMixtureCoeffs();
    
    /**
     * @brief Add offset to the diagonal of the covariance matrices
     * @details Guarantees convergence through covariance matrix invertibility
     */
    void addCovarianceOffset();
    
    /**
     * @brief Update inverse covariances of each Gaussian component
     * @throws runtime_error if one of the covariance matrices is not invertible
     */
    void updateInverseCovariances();
    
    /*@}*/

#pragma mark > Performance
    /*@{*/
    /** @name Performance: protected methods */
    /**
     * @brief Compute likelihood and estimate components probabilities
     * @details If the model is bimodal, the likelihood is computed only on the input modality,
     * except if 'observation_output' is specified.
     * Updates the likelihood buffer used to smooth likelihoods.
     * @param observation observation vector (full size for unimodal, input modality for bimodal)
     * @param observation_output observation vector of the output modality
     */
    double likelihood(vector<float> const& observation, vector<float> const& observation_output = NULLVEC_FLOAT);
    
    /**
     * @brief Compute Gaussian Mixture Regression
     * @details Estimates the output modality using covariance-based regression weighted by components' likelihood
     * @warning the function does not estimates the likelihoods, use 'likelihood' before performing
     * the regression.
     * @param observation_input observation vector of the input modality
     * @param predicted_output observation vector where the predicted observation for the output
     * modality is stored.
     */
    void regression(vector<float> const& observation_input, vector<float>& predicted_output);
    
    /*@}*/

#pragma mark -
#pragma mark === Protected attributes ===
    /**
     * @brief Number of Gaussian Mixture Components
     */
    int nbMixtureComponents_;
    
    /**
     * @brief Offset Added to the diagonal of covariance matrices for convergence
     */
    float covarianceOffset_;
};


#endif
