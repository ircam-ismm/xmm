//
// probabilistic_model.h
//
// Machine learning model based on the EM algorithm
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#ifndef rtml_probabilistic_model_h
#define rtml_probabilistic_model_h

#include "ringbuffer.h"
#include "training_set.h"
#if __cplusplus > 199711L
#include <mutex>
#endif

using namespace std;

/**
 * @enum CALLBACK_FLAG
 * @brief Flags for the Callback called by the training algorithm
 */
enum CALLBACK_FLAG
{
    /**
     * Training is still running
     */
    TRAINING_RUN,
    
    /**
     * Training is done without error
     */
    TRAINING_DONE,
    
    /**
     * An error occured during training (probably convergence issue)
     */
    TRAINING_ERROR
};

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
    unsigned int minSteps;

    /**
     * @brief Maximum number of iterations of the EM algorithm.
     * @details If this value is superior to
     * minSteps, this criterion is used. Otherwise, only the 'percentChg' criterion applies.
     */
    unsigned int maxSteps;

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
 * @class ProbabilisticModel
 * @brief Generic Template for Machine Learning Probabilistic models based on the EM algorithm
 */
class ProbabilisticModel : public Listener
{
public:
    template<typename modelType> friend class ModelGroup;
    friend class GMMGroup;
    friend class HierarchicalHMM;
    
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
    ProbabilisticModel(rtml_flags flags = NONE, TrainingSet *trainingSet = NULL);
    
    /**
     * @brief Copy Constructor
     * @param src Source Model
     */
    ProbabilisticModel(ProbabilisticModel const& src);
    
    /**
     * @brief Assignment
     * @param src Source Model
     */
    ProbabilisticModel& operator=(ProbabilisticModel const& src);
    
    /**
     * @brief Destructor
     */
    virtual ~ProbabilisticModel();
    /*@}*/
    
#pragma mark > Accessors
    /*@{*/
    /** @name Accessors */
    /**
     * @brief set the training set associated with the model
     * @details updates the dimensions of the model
     * @param trainingSet pointer to the training set.
     * @throws runtime_error if the training set has not the same number of modalities
     */
    void set_trainingSet(TrainingSet *trainingSet);
    
    /**
     * @brief Get Total Dimension of the model (sum of dimension of modalities)
     * @return total dimension of Gaussian Distributions
     */
    unsigned int get_dimension() const;
    
    /**
     * @brief Get the dimension of the input modality
     * @warning This can only be used in bimodal mode (construction with 'BIMODAL' flag)
     * @return dimension of the input modality
     * @throws runtime_error if not in bimodal mode
     */
    unsigned int get_dimension_input() const;
    
    /**
     * @brief Get minimum number of EM steps
     * @return minimum number of steps of the EM algorithm
     */
    unsigned int get_EM_minSteps() const;
    
    /**
     * @brief Get maximum number of EM steps
     * @return maximum number of steps of the EM algorithm
     * @see EMStopCriterion
     */
    unsigned int get_EM_maxSteps() const;
    
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
    void set_EM_minSteps(unsigned int steps);
    
    /**
     * @brief Set maximum number of steps of the EM algorithm
     * @param steps maximum number of steps of the EM algorithm
     * @throws invalid_argument if steps < 1
     */
    void set_EM_maxSteps(unsigned int steps);
    
    /**
     * @brief Set convergence threshold in percent-change of the likelihood
     * @param logLikelihoodPercentChg log-likelihood percent-change convergence threshold
     * @throws invalid_argument if logLikelihoodPercentChg <= 0
     */
    void set_EM_percentChange(double logLikelihoodPercentChg);
    
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
    
    /**
     * @brief set the callback function associated with the training algorithm
     * @details the function is called whenever the training is over or an error happened during training
     */
    void set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata);
    
    /*@}*/
    
#pragma mark > Performance
    /*@{*/
    /** @name Performance */
    /**
     * @brief Initialize the 'Performance' phase: prepare model for performance.
     */
    virtual void performance_init();
    
    /**
     * @brief Main Play function: updates the predictions of the model given a new observation
     * @param observation observation vector (must be of size 'dimension' or 'dimension_input'
     * depending on the mode [unimodal/bimodal])
     * @return likelihood of the observation
     */
    virtual double performance_update(vector<float> const& observation) = 0;
    
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
     * Pointer to the training set.
     */
    TrainingSet *trainingSet;
    
    /**
     * defines if the model is trained
     */
    bool trained;
    
    /**
     * progression within the training algorithm
     * @warning  not yet implemented?
     * @todo  implement training progression
     */
    float trainingProgression;
    
    /**
     * @brief Results: Instantaneous likelihood
     */
    double results_instant_likelihood;
    
    /**
     * @brief Results: Cumulative log-likelihood computed on a sliding window
     */
    double results_log_likelihood;
    
    /**
     * @brief Results: Predicted sound parameters (only used in regression mode)
     * @warning this variable is not allocated if the model is not bimodal
     */
    vector<float> results_predicted_output;
    
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
    virtual void _copy(ProbabilisticModel *dst, ProbabilisticModel const& src);
    /*@}*/
    
    /*@{*/
    /** @name Utility */
    /**
     * @brief update the content of the likelihood buffer and return average likelihood.
     * @details The method also updates the cumulative log-likelihood computed over a window (cumulativeloglikelihood)
     * @param instantLikelihood instantaneous likelihood at the current step
     * @return mean of the likelihood buffer
     */
    void updateLikelihoodBuffer(double instantLikelihood);
    
    /**
     * @brief handle notifications of the training set
     * @details here only the dimensions attributes of the training set are considered
     * @param attribute name of the attribute: should be either "dimension" or "dimension_input"
     */
    void notify(string attribute);
    
    /**
     * @brief Allocate memory for the model's parameters
     * @details called when dimensions are modified
     */
    virtual void allocate() = 0;
    
    /*@}*/
    
    /*@{*/
    /** @name Training: internal methods */
    /**
     * @brief Initialize the training algorithm
     */
    virtual void train_EM_init() = 0;
    
    /**
     * @brief Update Method of the EM algorithm
     * @details performs E and M steps of the EM algorithm.
     * @return likelihood of the training data given the current model parameters (before re-estimation).
     */
    virtual double train_EM_update() = 0;
    
    /**
     * @brief Terminate the training algorithm
     */
    virtual void train_EM_terminate();
    
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
     * @brief Construction flags
     */
    rtml_flags flags_;
    
    /**
     * @brief defines if the phrase is bimodal (true) or unimodal (false)
     */
    bool bimodal_;
    
    /**
     * @brief Total dimension of the data (both modalities if bimodal)
     */
    unsigned int dimension_;
    
    /**
     * @brief Dimension of the input modality
     */
    unsigned int dimension_input_;
    
    /**
     * @brief Callback function for the training algorithm
     */
    void (*trainingCallback_)(void *srcModel, CALLBACK_FLAG state, void* extradata);
    
    /**
     * @brief Extra data to pass in argument to the callback function
     */
    void *trainingExtradata_;
    
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