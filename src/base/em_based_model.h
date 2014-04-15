//
// em_based_learning_model.h
//
// Machine learning model based on the EM algorithm
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#ifndef rtml_em_based_learning_model_h
#define rtml_em_based_learning_model_h

#include "base_model.h"
#include "ringbuffer.h"
#if __cplusplus > 199711L
#include <mutex>
#endif

using namespace std;

/**
 * Default value for the minimum number of steps of the EM algorithm
 */
const int EM_MODEL_DEFAULT_EMSTOP_MINSTEPS = 10;

/**
 * Default value for the maximum number of steps of the EM algorithm
 */
const int EM_MODEL_DEFAULT_EMSTOP_MAXSTEPS = 0;

/**
 * Default value for the percent-change criterion of the EM algorithm
 */
const double EM_MODEL_DEFAULT_EMSTOP_PERCENT_CHG = 0.01;

/**
 * Default value for the size of the likelihood window used for smoothing likelihoods
 * in performance mode.
 */
const int EM_MODEL_DEFAULT_LIKELIHOOD_WINDOW = 1;

/**
 * @brief Stop Criterion for the EM algorithm
 */
struct EMStopCriterion {
    /**
     * @brief Minimum number of iterations of the EM algorithm
     */
    int minSteps;

    /**
     * @brief Maximum number of iterations of the EM algorithm.
     * @details If this value is superior to
     * minSteps, this criterion is used. Otherwise, only the 'percentChg' criterion applies.
     */
    int maxSteps;

    /**
     * @brief log-likelihood difference threshold to stop the EM algorithm. 
     * @details When the percent-change
     * in likelihood of the training data given the estimated parameters gets under this threshold,
     * the EM algorithm is stopped.
     */
    double percentChg;
};

/**
 * @ingroup ModelBase
 * @class EMBasedModel
 * @brief Generic Template for Machine Learning Probabilistic models based on the EM algorithm
 */
class EMBasedModel : public BaseModel
{
public:
    /**
     * @struct Results
     * @brief Results of the prediction of the model (recognition and/or regression)
     */
    struct Results {
        /**
         * Instantaneous likelihood
         */
        double instant_likelihood;
        
        /**
         * Cumulative log-likelihood computed on a sliding window
         */
        double logLikelihood;
        
        /**
         * Predicted sound parameters (only used in regression mode)
         * @warning this variable is not allocated if the model is not bimodal
         */
        vector<float> predicted_output;
    };

#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Constructors
    /*@{*/
    /** @name Constructors*/
    /**
     * @brief Constructor
     * @param trainingSet training set associated with the model
     * @param flags Construction Flags. The only valid flag here is BIMODAL, that defines if the
     * model is bimodal (can be used for regression).
     */
    EMBasedModel(rtml_flags flags = NONE, TrainingSet *trainingSet = NULL);
    
    /**
     * @brief Copy Constructor
     * @param src Source Model
     */
    EMBasedModel(EMBasedModel const& src);
    
    /**
     * @brief Assignment
     * @param src Source Model
     */
    EMBasedModel& operator=(EMBasedModel const& src);
    
    /**
     * @brief Destructor
     */
    virtual ~EMBasedModel();
    
    /*@}*/

#pragma mark > Training
    /*@{*/
    /** @name Training */
    /**
     * @brief Main training method based on the EM algorithm
     * @details the method performs a loop over the pure virtual method train_EM_update() until convergence.
     * The @a train_EM_update method computes both E and M steps of the EM algorithm.
     * @see train_EM_update
     */
    int train();
    
    /*@}*/

#pragma mark > EM Stop Criterion
    /*@{*/
    /** @name EM Stop Criterion */
    /**
     * @brief Get minimum number of EM steps
     * @return minimum number of steps of the EM algorithm
     */
    int get_EM_minSteps() const;
    
    /**
     * @brief Get maximum number of EM steps
     * @return maximum number of steps of the EM algorithm
     * @see EMStopCriterion
     */
    int get_EM_maxSteps() const;
    
    /**
     * @brief Get EM convergence threshold in percent-change of the likelihood
     * @return loglikelihood percent-change convergence threshold
     * @see EMStopCriterion
     */
    double get_EM_percentChange() const;
    
    /**
     * @brief Set minimum number of steps of the EM algorithm
     * @param steps minimum number of steps of the EM algorithm
     * @throws invalid_argument if steps < 1
     */
    void set_EM_minSteps(int steps);
    
    /**
     * @brief Set maximum number of steps of the EM algorithm
     * @param steps maximum number of steps of the EM algorithm
     * @throws invalid_argument if steps < 1
     */
    void set_EM_maxSteps(int steps);
    
    /**
     * @brief Set convergence threshold in percent-change of the likelihood
     * @param logLikelihoodPercentChg log-likelihood percent-change convergence threshold
     * @throws invalid_argument if logLikelihoodPercentChg <= 0
     */
    void set_EM_percentChange(double logLikelihoodPercentChg);
    
    /*@}*/

#pragma mark > Likelihood Buffer
    /*@{*/
    /** @name Likelihood Buffer */
    /**
     * @brief get size of the likelihood smoothing buffer (number of frames)
     * @return size of the likelihood smoothing buffer
     */
    unsigned int get_likelihoodBufferSize() const;
    
    /**
     * @brief set size of the likelihood smoothing buffer (number of frames)
     * @param likelihoodBufferSize size of the likelihood smoothing buffer
     * @throws invalid_argument if likelihoodBufferSize is < 1
     */
    void set_likelihoodBufferSize(unsigned int likelihoodBufferSize);
    
    /**
     * @brief update the content of the likelihood buffer and return average likelihood.
     * @details The method also updates the cumulative log-likelihood computed over a window (cumulativeloglikelihood)
     * @param instantLikelihood instantaneous likelihood at the current step
     * @return mean of the likelihood buffer
     */
    void updateLikelihoodBuffer(double instantLikelihood);
    
    /**
     * @brief Initialize the 'Performance' phase: prepare model for playing.
     */
    virtual void initPlaying();
    
    /*@}*/

#pragma mark > JSON I/O
    /*@{*/
    /** @name JSON I/O */
    /**
     * @brief Write to JSON Node
     * @return JSON Node containing training set information and data
     */
    virtual JSONNode to_json() const;
    
     /**
     * @brief Read from JSON Node
     * @param root JSON Node containing training set information and data
     * @throws JSONException if the JSON Node has a wrong format
     */
    virtual void from_json(JSONNode root);
    
    /*@}*/

#pragma mark -
#pragma mark === Public Attributes ===
    /**
     * @brief Results of the model
     */
    Results results;
    
protected:
#pragma mark -
#pragma mark === Protected Methods ===
    /*@{*/
    /** @name Copy between models */
    /**
     * @brief Copy between two models
     * @param src Source Model
     * @param dst Destination Model
     */
    using BaseModel::_copy;
    virtual void _copy(EMBasedModel *dst, EMBasedModel const& src);
    
    /*@}*/

#pragma mark > Training
    /*@{*/
    /** @name Training: pure virtual methods */
    /**
     * @brief Update Method of the EM algorithm
     * @details performs E and M steps of the EM algorithm.
     * @return likelihood of the training data given the current model parameters (before re-estimation).
     */
    virtual double train_EM_update() = 0;
    
    /**
     * @brief checks if the training has converged according to the object's EM stop criterion
     * @param step index of the current step of the EM algorithm
     * @param log_prob log-likelihood returned by the EM update
     * @param old_log_prob log-likelihood returned by the EM update at the previous step
     */
    bool train_EM_stop(int step, double log_prob, double old_log_prob) const;
    
    /*@}*/

#pragma mark -
#pragma mark === Protected Attributes ===
    /**
     * @brief Likelihood buffer used for smoothing
     */
    RingBuffer<double, 1> likelihoodBuffer_;

    /**
     * @brief Stop criterion of the EM algorithm
     * @see EMStopCriterion
     */
    EMStopCriterion stopcriterion_;
    
#if __cplusplus > 199711L
    /**
     * @brief Mutex used in Concurrent Mode
     */
    mutex trainingMutex;
#endif
};

#endif