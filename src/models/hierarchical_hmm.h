//
// hierarchical_hmm.h
//
// Hierarchical Hidden Markov Model
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
//

#ifndef mhmm_hierarchical_hmm_h
#define mhmm_hierarchical_hmm_h

#include "model_group.h"
#include "hmm.h"

using namespace std;

/**
 * @brief Default exit transition for the highest level
 */
const double HHMM_DEFAULT_EXITTRANSITION = 0.1;

/**
 * @brief Default value for the Incremental Learning update of the transition matrix
 */
const bool HHMM_DEFAULT_INCREMENTALLEARNING = false;

/**
 * @brief Default regularization factor for the update of transition
 * probabilities for incremental learning
 */
const bool HHMM_DEFAULT_REGULARIZATIONFACTOR = false;

/**
 * @ingroup HMM
 * @class HierarchicalHMM
 * @brief Hierarchical Hidden Markov Model
 * @todo Write detailed documentation
 */
class HierarchicalHMM : public ModelGroup< HMM > {
public:
#pragma mark -
#pragma mark === Public Interface ===
    /**
     * @brief Iterator over models
     */
    typedef typename  map<Label, HMM>::iterator model_iterator;
    
    /**
     * @brief Constant Iterator over models
     */
    typedef typename  map<Label, HMM>::const_iterator const_model_iterator;
    
#pragma mark > Constructor
    /*@{*/
    /** @name Constructor */
    HierarchicalHMM(rtml_flags flags = NONE,
                    TrainingSet *_globalTrainingSet=NULL);
    
    virtual ~HierarchicalHMM();
    
    /*@}*/

#pragma mark 
    /*@{*/
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
    
    bool get_estimateMeans() const;
    
    void set_estimateMeans(bool _estimateMeans);
    
    void addExitPoint(int state, float proba);
    
    /**
     * @brief return learning mode: "incremental" or "ergodic"
     * @details  if ergodic, each time a model is added at the high level, the transition
     * matrix is reset to ergodic. if "incremental", the transitions are updated using regularization
     * @return learningMode "incremental" or "ergodic"
     */
    string get_learningMode() const;
    
    /**
     * @brief set learning mode: "incremental" or "ergodic"
     * @details  if ergodic, each time a model is added at the high level, the transition
     * matrix is reset to ergodic. if "incremental", the transitions are updated using regularization
     * @param learningMode "incremental" or "ergodic"
     * @throws invalid_argument if the argument is neither "incremental" nor "ergodic"
     */
    void set_learningMode(string learningMode);
    
    /**
     * @brief get a copy of the high-level Prior probabilities vector
     * @return High-level prior probability vector
     * @warning memory is allocated for the returned array (need to be freed)
     */
    double* get_prior() const;
    
    /**
     * @brief set high-level prior probabilities vector
     * @param prior high-level probability vector (size nbPrimitives)
     * @throws invalid_argument if the array has a wrong format (not enough values)
     * @warning the models are ordered in ascending order by label
     */
    void set_prior(double *prior);
    
    /**
     * @brief get a copy of the high-level transition matrix
     * @return High-level transition matrix
     * @warning memory is allocated for the returned array (need to be freed)
     */
    double* get_transition() const;
    
    /**
     * @brief set the high-level  transition matrix
     * @param trans high-level transition matrix
     * @throws invalid_argument if the array has a wrong format (not enough values)
     * @warning the models are ordered in ascending order by label
     */
    void set_transition(double *trans);
    
    /**
     * @brief get a copy of the high-level exit probabilities
     * @details exit probabilities are the probabilities to finish and go
     * back to the root
     * @return a copy of the exit transition vector of the high level
     * @warning memory is allocated for the returned array (need to be freed)
     */
    double* get_exitTransition() const;
    
    /**
     * @brief set the exit transition vector of the high level
     * @param exittrans high-level exit probabilities vector
     * @throws invalid_argument if the array has a wrong format (not enough values)
     * @warning the models are ordered in ascending order by label
     */
    void set_exitTransition(double *exittrans);
    
    /**
     * @brief set a particular value of the transition matrix
     * @details sets trans(i,j) = proba
     * @param srcSegmentLabel origin segment
     * @param dstSegmentLabel target segment
     * @param proba probability of making a transition from srcSegmentLabel to dstSegmentLabel
     * @warning transitions are normalized after the value is set.
     * @todo absolute/relative mode?
     */
    void setOneTransition(Label srcSegmentLabel, Label dstSegmentLabel, double proba);

    /*@}*/
    
#pragma mark > Training
    /*@{*/
    /** @name Training */
    /**
     * @brief Remove Specific model
     * @details The method updates the transition parameters
     * @param label label of the model
     * @throw out_of_range if the label does not exist
     */
    virtual void remove(Label const& label);
    
    /*@}*/

#pragma mark > Performance
    /*@{*/
    /** @name Performance */
    /**
     * @brief Initialize performance mode
     */
    virtual void initPlaying();
    
    /**
     * @brief Main performance Function: perform joint recognition and mapping 
     * (in the case of a bimodal model)
     * @param observation observation vector. If the model is bimodal, this should be allocated for
     * both modalities, and should contain the observation on the input modality. The predicted
     * output will be appended to the input modality observation
     * @param modelLikelihoods output: likelihood of each model
     */
    virtual void play(float *observation, double *modelLikelihoods);
    
    /**
     * @brief Get the Results of a specific model
     * @param label label of the model
     * @return Results estimated by the model by the latest call at the play function
     * @todo this results stuff is crappy
     */
    HMM::Results getResults(Label const& label) const;
    
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


#ifdef SWIGPYTHON
#pragma mark > Python
    /*@{*/
    /** @name Python */
    void play(int dimension_, double *observation,
              int nbModels_, double *likelihoods,
              int nbModels__, double *cumulativelikelihoods)
    {
        int dimension = this->referenceModel.get_dimension();
        
        float *obs = new float[dimension];
        for (int d=0; d<dimension; d++) {
            obs[d] = float(observation[d]);
        }
        
        this->play(obs, likelihoods);
        
        int m(0);
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            cumulativelikelihoods[m++] = it->second.cumulativeloglikelihood;
        
        delete[] obs;
    }
    
    /**
     * @brief set high-level prior probabilities vector
     * @param nbPrimitives number of models
     * @param prior high-level probability vector (size nbPrimitives)
     * @throws invalid_argument if the array has a wrong format (not enough values)
     * @warning the models are ordered in ascending order by label
     */
    void set_prior(int nbPrimitives, double *prior_) {
        if (nbPrimitives != this->size())
            throw RTMLException("Prior vector: wrong size");
        this->set_prior(prior_);
    }
    
    /**
     * @brief set the high-level  transition matrix
     * @param nbPrimitivesSquared square of the number of models
     * @param trans high-level transition matrix
     * @throws invalid_argument if the array has a wrong format (not enough values)
     * @warning the models are ordered in ascending order by label
     */
    void set_transition(int nbPrimitivesSquared, double *trans_) {
        if (nbPrimitivesSquared != this->size()*this->size())
            throw RTMLException("Transition matrix: wrong size");
        this->set_transition(trans_);
    }
    /*@}*/
#endif
    
#pragma mark -
#pragma mark === Public attributes ===
    /**
     * @brief Prior probabilities of the models
     */
    map<Label, double> prior;
    
    /**
     * @brief exit probabilities of the model (probability to finish and go back to the root)
     */
    map<Label, double> exitTransition;
    
    /**
     * @brief Transition probabilities between models
     */
    map<Label, map<Label, double> > transition;
    
protected:
#pragma mark -
#pragma mark === Protected Methods ===
#pragma mark > High level parameters: update and estimation
    /*@{*/
    /** @name High level parameters: update and estimation */
    /**
     * @brief update high-level parameters when a new primitive is learned
     * @details  updated parameters: prior probabilities + transition matrix
     */
    void updateTransitionParameters();
    
    /**
     * @brief incremental learning: update high-level prior probabilities (regularization)
     */
    void updatePrior_incremental();
    
    /**
     * @brief incremental learning: update high-level transition matrix (regularization)
     * @details transition probabilities between primitive gestures
     */
    void updateTransition_incremental();
    
    /**
     * @brief ergodic learning update high-level prior probabilities -> equal prior probs
     */
    void updatePrior_ergodic();
    
    /**
     * @brief ergodic learning: update high-level transition matrix
     * @details equal transition probabilities between primitive gestures
     */
    void updateTransition_ergodic();
    
    /**
     * @brief Update exit probabilities of each sub-model
     */
    void updateExitProbabilities();
    
    virtual void updateTrainingSet(Label const& label);
    
    /**
     * @brief Normalize segment level prior and transition matrices
     */
    void normalizeTransitions();
    
    /*@}*/

#pragma mark > Forward Algorithm
    /*@{*/
    /** @name Forward Algorithm */
    /**
     * @brief TODO
     * @todo doc this
     */
    model_iterator forward_init(const float* observation, double* modelLikelihoods);
    
    /**
     * @brief TODO
     * @todo doc this
     */
    model_iterator forward_update(const float* observation, double* modelLikelihoods);
    
    /**
     * @brief get instantaneous likelihood
     *
     * get instantaneous likelihood on alpha variable for exit state exitNum.
     * @param exitNum number of exit state (0=continue, 1=transition, 2=back to root). if -1, get likelihood over all exit states
     * @param likelihoodVector likelihood vector (size nbPrimitives)
     */
    void likelihoodAlpha(int exitNum, vector<double> &likelihoodVector) const;

    /*@}*/

#pragma mark -
#pragma mark === Protected Attributes ===
    /**
     * Learning mode: if true, incremental learning is used.
     */
    bool incrementalLearning_;
    
    /**
     * @brief Defines if the forward algorithm has been initialized
     */
    bool forwardInitialized_;
    
    /**
     * @brief intermediate Forward variable (used in Frontier algorithm)
     */
    vector<double> V1_;
    
    /**
     * @brief intermediate Forward variable (used in Frontier algorithm)
     */
    vector<double> V2_;
};


#endif