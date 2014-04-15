//
// hmm.h
//
// Hidden Markov Model: Possibly Multimodal and/or submodel of a hierarchical model
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
// 

#ifndef mhmm_hmm_h
#define mhmm_hmm_h

#include "gmm.h"

using namespace std;

const int HMM_DEFAULT_NB_STATES = 10;
const bool HMM_DEFAULT_ESTIMATEMEANS = true;

const float PLAY_EM_MAX_LOG_LIK_PERCENT_CHG = 0.001;
const float PLAY_EM_STEPS = 5;

const float HMM_DEFAULT_EXITPROBABILITY_LAST_STATE = 0.1;

/**
 * @enum TRANSITION_MODE
 * @brief Mode of transition of the HMM
 */
typedef enum _TRANSITION_MODE {
    /**
     * @brief Ergodic Transition Matrix
     * @todo  remove this? And simplify algorithm for left-right matrix?
     */
    ERGODIC,
    
    /**
     * @brief Left-Right Transition model
     * @details  The only authorized transitions are: auto-transition and transition to the next state
     */
    LEFT_RIGHT
} TRANSITION_MODE;

/**
 * @class HMM
 * @brief Hidden Markov Model
 * @details Support Hierarchical Model: if built with the flag 'HIERARCHICAL', the model includes exit
 * transition probabilities. The model can be eith unimodal, or Multimodal when constructed with the flag 'BIMODAL'.
 */
class HMM : public EMBasedLearningModel
{
    friend class ConcurrentHMM;
    friend class HierarchicalHMM;
    
public:
    /**
     * @struct Results
     * @brief Structure containing the results of the recognition using the HMM.
     */
    typedef struct _ResultsHMM {
        /**
         * @brief Estimated time progression.
         * @details The time progression is computed as the centroid of the state
         * probability distribution estimated by the forward algorithm
         */
        double progress;
        
        /**
         * @brief Likelihood to exit the gesture on the next time step
         */
        double exitLikelihood;
        
        /**
         * @brief Likelihood of the gesture normalized over all gestures
         */
        double likelihoodnorm;
    } ResultsHMM;
    
    /**
     * @brief Iterator over the phrases of the associated training set
     */
    typedef typename map<int, Phrase* >::iterator phrase_iterator;
    
#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Constructors
    /** @name Constructors */
    /**
     * @brief Constructor
     * @param trainingSet Training set associated with the model
     * @param flags Construction Flags: use 'BIMODAL' for use with Regression (multimodal HMM). Use 'HIERARCHICAL'
     * for use as a submodel of a hierarchical HMM.
     * @param nbStates number of hidden states
     * @param nbMixtureComponents number of gaussian mixture components for each state
     */
    HMM(rtml_flags flags = NONE,
        TrainingSet *trainingSet = NULL,
        int nbStates = HMM_DEFAULT_NB_STATES,
        int nbMixtureComponents = GMM_DEFAULT_NB_MIXTURE_COMPONENTS);
    
    /**
     * @brief Copy constructor
     * @param src Source Model
     */
    HMM(HMM const& src);
    
    /**
     * @brief Assignment
     * @param src Source Model
     */
    HMM& operator=(HMM const& src);
    
    /**
     * @brief Destructor
     */
    virtual ~HMM();
    
#pragma mark > Accessors
    /** @name Accessors */
    /**
     * @brief Get the Number of hidden states of the model
     * @return number of hidden states
     */
    virtual int get_nbStates() const;

    /**
     * @brief Set the number of hidden states of the model
     * @details sets the mode to be untrained
     * @param nbStates number of hidden states
     * @throws invalid_argument if the number of states is <= 0
     */
    void set_nbStates(int nbStates);
    
    /**
     * @brief Get the number of Gaussian mixture components of the observation probability distribution
     * @return number of Gaussian mixture components
     */
    int get_nbMixtureComponents() const;
    
    /**
     * @brief Set the number of Gaussian mixture components of the observation probability distribution
     * @param nbMixtureComponents number of Gaussian mixture components
     * @throws invalid_argument if the number of Gaussian mixture components is <= 0
     */
    void set_nbMixtureComponents(int nbMixtureComponents);
    
    /**
     * @brief Get the offset added to the diagonal of covariance matrices for convergence
     * @return offset added to the diagonal of covariance matrices
     */
    float  get_covarianceOffset() const;
    
    /**
     * @brief Get the offset added to the diagonal of covariance matrices for convergence
     * @param covarianceOffset offset added to the diagonal of covariance matrices
     */
    void set_covarianceOffset(float covarianceOffset);
    
    /**
     * @brief get transition mode of the hidden Markov Chain
     * @return string corresponding to the transition mode (left-right / ergodic)
     * @todo: remove transitionMode to simplify forward complexity
     */
    string get_transitionMode() const;
    
    /**
     * @brief set transition mode of the hidden Markov Chain
     * @param transMode_str string keyword corresponding to the transition mode ("left-right" / "ergodic")
     * @throws invalid_argument if the argument is not "left-right" or "ergodic"
     */
    void set_transitionMode(string transMode_str);
    
#pragma mark > Observation probabilities
    /** @name Observation probabilities */
    /**
     * @brief Gaussian observation probability of a given state
     * @param observation observation vector
     * @param stateIndex index of the state
     * @param mixtureComponent index of the Gaussian mixture component (full mixture observation probability if unspecified)
     * @return likelihood of the observation for state stateIndex given the model parameters
     * @throws out_of_range if the index of the state is out of range
     * @throws out_of_range if the index of the Gaussian Mixture Component is out of bounds
     * @throws runtime_error if a Covariance Matrix is not invertible
     */
    double obsProb(const float *observation, unsigned int stateIndex, int mixtureComponent=-1);
    
    /**
     * @brief Gaussian observation probability of a given state for the input modality
     * @param observation_input observation vector of the input modality
     * @param stateIndex index of the state
     * @param mixtureComponent index of the Gaussian mixture component (full mixture observation probability if unspecified)
     * @return likelihood of the observation for state stateIndex given the model parameters
     * @throws runtime_error if the model is not bimodal
     * @throws out_of_range if the index of the state is out of range
     * @throws out_of_range if the index of the Gaussian Mixture Component is out of bounds
     * @throws runtime_error if a Covariance Matrix of the input modality is not invertible
     */
    double obsProb_input(const float *observation_input, unsigned int stateIndex, int mixtureComponent=-1);
    
    /**
     * @brief Gaussian observation probability of a given state (bimodal model)
     * @param observation_input observation vector of the input modality
     * @param observation_output observation vector of the output modality
     * @param stateIndex index of the state
     * @param mixtureComponent index of the Gaussian mixture component (full mixture observation probability if unspecified)
     * @return likelihood of the observation for state stateIndex given the model parameters
     * @throws runtime_error if the model is not bimodal
     * @throws out_of_range if the index of the state is out of range
     * @throws out_of_range if the index of the Gaussian Mixture Component is out of bounds
     * @throws runtime_error if a Covariance Matrix of the input modality is not invertible
     */
    double obsProb_bimodal(const float *observation_input,
                           const float *observation_output,
                           unsigned int stateIndex,
                           int mixtureComponent=-1);
    
#pragma mark > Play!
    /** @name Playing */
    /**
     * @brief Initialize the 'Performance' phase
     */
    void initPlaying();

    /**
     * @brief Main Play function: performs recognition (unimodal mode) or regression (bimodal mode)
     * @details The predicted output is stored in the observation vector in bimodal mode
     * @param observation pointer to current observation vector. Must be of size 'dimension' (input + output dimension).
     * @return likelihood computed on the gesture modality by a forward algorithm
     */
    double play(float *observation);
    
    /**
     * @brief Get the results structure
     * @return Results structure
     */
    Results getResults();
    
#pragma mark > JSON I/O
    /** @name JSON I/O */
    /**
     * @brief Write to JSON Node
     * @return JSON Node containing model information and parameters
     */
    virtual JSONNode to_json() const;
    
    /**
     * @brief Read from JSON Node
     * @details allocate model parameters and updates inverse Covariances
     * @param root JSON Node containing model information and parameters
     * @throws JSONException if the JSONNode has a wrong format
     */
    virtual void from_json(JSONNode root);
    
#pragma mark > Exit Probabilities
    /**
     * @brief Set the exit probability of a specific state
     * @details this method is only active in 'HIERARCHICAL' mode. The probability
     * @param stateIndex index of the state to add the exit point
     * @param proba probability to exit the gesture from this state
     * @throws runtime_error if the model is not hierarchical
     * @throws out_of_range if the state index is out of bounds
     */
    void addExitPoint(int stateIndex, float proba);
    
#pragma mark > Python
    /** @name Python methods */
#ifdef SWIGPYTHON
    /**
     * @brief Python Binding for the play function (to use with SWIG)
     * @param dimension_ dimension of the observation vector (both modalities if BIMODAL)
     * @param observation observation vector
     * @param nbStates_ number of hidden states
     * @param alpha_ used to store state probabilities after forward update
     * @return lieklihood of the observation vector given the model and past observations.
     */
    double play(int dimension_, double *observation,
                int nbStates_, double *alpha_);
#endif
    
#pragma mark -
#pragma mark === Public Attributes ===
    /**
     * @brief Results estimated by the model
     * @details These results are updated for each new observation in playing mode
     */
    ResultsHMM results_hmm;
    
    /**
     * @brief State probabilities estimated by the forward algorithm.
     */
    vector<double> alpha;
    
    /**
     * @brief State probabilities estimated by the hierarchical forward algorithm.
     * @details the variable is only allocated/used in hierarchical mode (see 'HIERARCHICAL' flag)
     */
    vector<double> alpha_h[3];

    protected:
#pragma mark -
#pragma mark === Protected Methods ===
    /**
     * @brief Copy between 2 MHMM models (called by copy constructor and assignment methods)
     * @param src Source Model
     * @param dst Destination Model
     */
    using EMBasedLearningModel::_copy;
    virtual void _copy(HMM *dst, HMM const& src);
    
#pragma mark > Parameters initialization
    /** @name Parameters initialization */
    /**
     * @brief Allocate model parameters
     */
    void allocate();
    
    /**
     * @brief Evaluate the number of hidden states based on the length of the training examples
     * @todo integrate state factor as attribute in HMM class
     */
    void evaluateNbStates(int factor = 5);
    
    /**
     * @brief initialize model parameters to their default values
     */
    virtual void initParametersToDefault();
    
    /**
     * @brief initialize the means of each state with the first phrase (single gaussian)
     */
    void initMeansWithFirstPhrase();
    
    /**
     * @brief initialize the means of each state with all training phrases (single gaussian)
     */
    void initMeansWithAllPhrases_single();
    
    /**
     * @brief initialize the covariances of each state with all training phrases (single gaussian)
     */
    void initCovariancesWithAllPhrases_single();
    
    /**
     * @brief initialize the means of each states with all training phrases (mixture of gaussian)
     */
    void initMeansWithAllPhrases_mixture();
    
    /**
     * @brief initialize the covariances of each states with all training phrases (mixture of gaussian)
     */
    void initCovariancesWithAllPhrases_mixture();
    
    /**
     * @brief set the prior and transition matrix to ergodic
     */
    void setErgodic();
    
    /**
     * @brief set the prior and transition matrix to left-right (no state skip)
     */
    void setLeftRight();
    
    /**
     * @brief Normalize transition probabilities
     */
    void normalizeTransitions();
    
#pragma mark > Forward-Backward algorithm
    /** @name Forward-Backward Algorithm */
    /**
     * @brief Initialization of the forward algorithm
     * @param observation observation vector (input modalit if bimodal)
     * @param observation observation vector for the output modality. If undefined, the algorithm
     * is only ran on the input modality.
     * @return likelihood of the observation
     */
    double forward_init(const float *observation, const float *observation_output=NULL);
    
    /**
     * @brief Update of the forward algorithm
     * @param observation observation vector (input modalit if bimodal)
     * @param observation observation vector for the output modality. If undefined, the algorithm
     * @return likelihood
     */
    double forward_update(const float *observation, const float *observation_output=NULL);
    
    /**
     @brief Forward update with the estimated output observation
     @deprecated generally unused in current version of max/python implementations
     */
    double forward_update_withNewObservation(const float *observation, const float *observation_output);
    
    /**
     * @brief Initialization Backward algorithm
     * @param ct inverse of the likelihood at time step t computed
     * with the forward algorithm (see Rabiner 1989)
     */
    void backward_init(double ct);
    
    /**
     * @brief Update of the Backward algorithm
     * @param observation observation vector at time t
     * @param ct inverse of the likelihood at time step t computed
     * with the forward algorithm (see Rabiner 1989)
     */
    void backward_update(double ct, const float *observation, const float *observation_output = NULL);
    
#pragma mark > Training algorithm
    /** @name Training Algorithm */
    /**
     * @brief Initialization of the parameters before training
     */
    void initTraining();
    
    /**
     * @brief Termination of the training algorithm
     */
    void finishTraining();
    
    /**
     * @brief update method of the EM algorithm (calls baumWelch_update)
     */
    virtual double train_EM_update();
    /**
     * @brief Baum-Welch update for Hidden Markov Models
     */
    double baumWelch_update();
    
    /**
     * @brief Compute the forward-backward algorithm on a phrase of the training set
     * @param currentPhrase pointer to the phrase of the training set
     * @param phraseIndex index of the phrase
     * @return lieklihood of the phrase given the model's current parameters
     */
    double baumWelch_forwardBackward(Phrase* currentPhrase, int phraseIndex);
    
    void baumWelch_gammaSum();
    
    /**
     * @brief Estimate the Coefficiant of the Gaussian Mixture for each state
     */
    void baumWelch_estimateMixtureCoefficients();
    
    /**
     * @brief Estimate the Means of the Gaussian Distribution for each state
     */
    void baumWelch_estimateMeans();
    
    /**
     * @brief Estimate the Covariances of the Gaussian Distribution for each state
     */
    void baumWelch_estimateCovariances();
    
    /**
     * @brief Estimate the Prior Probabilities
     */
    void baumWelch_estimatePrior();
    
    /**
     * @brief Estimate the Transition Probabilities
     */
    void baumWelch_estimateTransitions();
    
#pragma mark > Play!
    /** @name Playing */
    /**
     * @brief Adds a cyclic Transition probability (from last state to first state)
     * @details avoids getting stuck at the end of the model. this method is idle for a hierarchical model.
     * @param proba probability of the transition form last to first state
     */
    void addCyclicTransition(double proba);
    
    void regression(float *observation_input, vector<float>& predicted_output);
    
    void updateTimeProgression();
    
#pragma mark > Exit Probabilities
    /**
     * @brief Update the exit probability vector given the probabilities
     * @details this method is only active in 'HIERARCHICAL' mode. The probability
     * vector defines the probability of exiting the gesture from each state. If unspecified,
     * only the last state of the gesture has a non-zero probability.
     * @param _exitProbabilities vector of exit probabilities (size must be nbStates)
     * @throws runtime_error if the model is not hierarchical
     */
    void updateExitProbabilities(float *_exitProbabilities = NULL);
    
#pragma mark -
#pragma mark === Protected Attributes ===
    /**
     * Number of hidden states of the Markov chain
     */
    int nbStates_;
    
    /**
     * Number of Gaussian mixture components for each state
     */
    int nbMixtureComponents_;
    
    /**
     * Offset added to the diagonal of covariance matrices for convergence
     */
    float  covarianceOffset_;
    
    /**
     * Transition mode of the model (left-right vs ergodic)
     */
    TRANSITION_MODE transitionMode_;
    
    /**
     * Prior probabilities
     */
    vector<float> prior_;
    
    /**
     * Transition Matrix
     * @todo make it smaller to be left-right specific
     */
    vector<float> transition_;
    
    /**
     * States of the model (Gaussian Mixture Models)
     */
    vector<GMM> states_;
    
    /**
     * Stop criterion for the EM estimation during performance
     * @deprecated this is not currently used
     */
    EMStopCriterion play_EM_stopCriterion_;
    
    /**
     * Defines if the forward algorithm has been initialized
     */
    bool forwardInitialized_;
    
    /**
     * Define if the EM algorithm should re-estimate the means of each state.
     */
    bool estimateMeans_;
    
    /**
     * used to store the alpha estimated at the previous time step
     */
    vector<double> previousAlpha_;
    
    /**
     * backward state probabilities
     */
    vector<double> beta_;
    
    /**
     * used to store the beta estimated at the previous time step
     */
    vector<double> previousBeta_;
    
    /**
     * Sequence of Gamma probabilities
     */
    vector< vector<double> > gammaSequence_;
    
    /**
     * Sequence of Epsilon probabilities
     */
    vector< vector<double> > epsilonSequence_;
    
    /**
     * Sequence of Gamma probabilities for each mixture component
     */
    vector< vector< vector<double> > > gammaSequencePerMixture_;
    
    /**
     * Sequence of alpha (forward) probabilities
     */
    vector<double> alpha_seq_;
    
    /**
     * Sequence of beta (backward) probabilities
     */
    vector<double> beta_seq_;
    
    vector<double> gammaSum_;
    vector<double> gammaSumPerMixture_;
    
    /**
     * Defines if the model is a submodel of a hierarchical HMM
     */
    bool is_hierarchical_;
    
    /**
     * Exit probabilities for a hierarchical model.
     */
    vector<float> exitProbabilities_;
};


#endif
