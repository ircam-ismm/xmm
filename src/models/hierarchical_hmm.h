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
    typedef map<Label, HMM>::iterator model_iterator;
    
    /**
     * @brief Constant Iterator over models
     */
    typedef map<Label, HMM>::const_iterator const_model_iterator;
    
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
    double  get_covarianceOffset() const;
    
    /**
     * @brief Get the offset added to the diagonal of covariance matrices for convergence
     * @param covarianceOffset offset added to the diagonal of covariance matrices
     */
    void set_covarianceOffset(double covarianceOffset);
    
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
    void get_prior(vector<double>& prior) const;
    
    /**
     * @brief set high-level prior probabilities vector
     * @param prior high-level probability vector (size nbPrimitives)
     * @throws invalid_argument if the array has a wrong format (not enough values)
     * @warning the models are ordered in ascending order by label
     */
    void set_prior(vector<double> const& prior);
    
    /**
     * @brief get a copy of the high-level transition matrix
     * @return High-level transition matrix
     * @warning memory is allocated for the returned array (need to be freed)
     */
    void get_transition(vector<double>& trans) const;
    
    /**
     * @brief set the high-level  transition matrix
     * @param trans high-level transition matrix
     * @throws invalid_argument if the array has a wrong format (not enough values)
     * @warning the models are ordered in ascending order by label
     */
    void set_transition(vector<double> const& trans);
    
    /**
     * @brief get a copy of the high-level exit probabilities
     * @details exit probabilities are the probabilities to finish and go
     * back to the root
     * @return a copy of the exit transition vector of the high level
     * @warning memory is allocated for the returned array (need to be freed)
     */
    void get_exitTransition(vector<double>& trans) const;
    
    /**
     * @brief set the exit transition vector of the high level
     * @param exittrans high-level exit probabilities vector
     * @throws invalid_argument if the array has a wrong format (not enough values)
     * @warning the models are ordered in ascending order by label
     */
    void set_exitTransition(vector<double> const& exittrans);
    
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
    virtual void performance_init();
    
    /**
     * @brief Main performance Function: perform joint recognition and mapping 
     * (in the case of a bimodal model)
     * @param observation observation vector. If the model is bimodal, this should be allocated for
     * both modalities, and should contain the observation on the input modality. The predicted
     * output will be appended to the input modality observation
     */
    virtual void performance_update(vector<float> const& observation);
    
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
    model_iterator forward_init(const float* observation);
    
    /**
     * @brief TODO
     * @todo doc this
     */
    model_iterator forward_update(const float* observation);
    
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