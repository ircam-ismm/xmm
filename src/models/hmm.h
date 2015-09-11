/*
 * hmm.h
 *
 * Hidden Markov Model for continuous recognition and regression
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


#ifndef xmm_lib_hmm_h
#define xmm_lib_hmm_h

#include "gmm.h"

namespace xmm
{
    /**
     * @defgroup HMM Hidden Markov Models
     */
    
    /** @ingroup HMM
     * @class HMM
     * @brief Hidden Markov Model
     * @details Support Hierarchical Model: if built with the flag 'HIERARCHICAL', the model includes exit
     * transition probabilities. The model can be eith unimodal, or Multimodal when constructed with the flag 'BIMODAL'.
     */
    class HMM : public ProbabilisticModel
    {
        friend class HMMGroup;
        friend class HierarchicalHMM;
        
    public:
        ///@cond DEVDOC
        
        static const int DEFAULT_NB_STATES = 10;
        static const bool DEFAULT_ESTIMATEMEANS = true;
        
        static const float DEFAULT_EXITPROBABILITY_LAST_STATE() { return 0.1; }
        static const float TRANSITION_REGULARIZATION() { return 1.0e-5; }
        
        ///@endcond
        
        /**
         * @enum TRANSITION_MODE
         * @brief Mode of transition of the HMM
         * @todo: Remove transitionMode to simplify forward complexity
         */
        enum TRANSITION_MODE {
            /**
             * @brief Ergodic Transition Matrix
             */
            ERGODIC,
            
            /**
             * @brief Left-Right Transition model
             * @details The only authorized transitions are: auto-transition and transition to the next state
             */
            LEFT_RIGHT
        };
        
        /**
         * @enum REGRESSION_ESTIMATOR
         * @brief Estimator for the regression with HMMs
         */
        enum REGRESSION_ESTIMATOR {
            /**
             * @brief The output is estimated by a weighted regression over all states
             */
            FULL,
            
            /**
             * @brief The output is estimated by a weighted regression over a window centered around
             * the likeliest state
             */
            WINDOWED,
            
            /**
             * @brief The output is estimated by a regression using the likeliest state only.
             */
            LIKELIEST
        };
    
        /**
         * @brief Iterator over the phrases of the associated training set
         */
        typedef std::map<int, Phrase* >::iterator phrase_iterator;
        
#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Constructors
        /** @name Constructors */
        ///@{
        
        /**
         * @brief Constructor
         * @param trainingSet Training set associated with the model
         * @param flags Construction Flags: use 'BIMODAL' for use with Regression (multimodal HMM). Use 'HIERARCHICAL'
         * for use as a submodel of a hierarchical HMM.
         * @param nbStates number of hidden states
         * @param nbMixtureComponents number of gaussian mixture components for each state
         * @param covariance_mode covariance mode (full vs diagonal)
         */
        HMM(xmm_flags flags = NONE,
            TrainingSet *trainingSet = NULL,
            int nbStates = DEFAULT_NB_STATES,
            int nbMixtureComponents = GMM::DEFAULT_NB_MIXTURE_COMPONENTS,
            GaussianDistribution::COVARIANCE_MODE covariance_mode = GaussianDistribution::FULL);
        
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
        
        ///@}
        
#pragma mark > Accessors
        /** @name Accessors */
        ///@{
        
        /**
         * @brief set the training set associated with the model
         * @details updates the training sets of each GMM
         * @param trainingSet pointer to the training set.
         * @throws runtime_error if the training set has not the same number of modalities
         */
        void set_trainingSet(TrainingSet *trainingSet);
        
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
         * @return offset added to the diagonal of covariance matrices (relative to data variance)
         */
        double  get_varianceOffset_relative() const;
        
        /**
         * @brief Get the offset added to the diagonal of covariance matrices for convergence
         * @return offset added to the diagonal of covariance matrices (minimum value)
         */
        double  get_varianceOffset_absolute() const;
        
        /**
         * @brief Get the offset added to the diagonal of covariance matrices for convergence
         * @param varianceOffset_relative offset added to the diagonal of covariances matrices (relative to data variance)
         * @param varianceOffset_absolute offset added to the diagonal of covariances matrices (minimum value)
         */
        void set_varianceOffset(double varianceOffset_relative, double varianceOffset_absolute);
        
        /**
         * @brief get the current covariance mode
         */
        GaussianDistribution::COVARIANCE_MODE get_covariance_mode() const;
        
        /**
         * @brief set the covariance mode
         * @param covariance_mode target covariance mode
         */
        void set_covariance_mode(GaussianDistribution::COVARIANCE_MODE covariance_mode);
        
        /**
         * @brief Get the regression estimator type
         * @return regression estimator type
         * @see REGRESSION_ESTIMATOR
         */
        REGRESSION_ESTIMATOR get_regression_estimator() const;
        
        /**
         * @brief Set the regression estimator type
         * @param regression_estimator type of estimator
         * @see REGRESSION_ESTIMATOR
         */
        void set_regression_estimator(REGRESSION_ESTIMATOR regression_estimator);
        
        /**
         * @brief get transition mode of the hidden Markov Chain
         * @return string corresponding to the transition mode (left-right / ergodic)
         */
        std::string get_transitionMode() const;
        
        /**
         * @brief set transition mode of the hidden Markov Chain
         * @param transMode_str string keyword corresponding to the transition mode ("left-right" / "ergodic")
         * @throws invalid_argument if the argument is not "left-right" or "ergodic"
         */
        void set_transitionMode(std::string transMode_str);
        
        /**
         * @brief Set the exit probability of a specific state
         * @details this method is only active in 'HIERARCHICAL' mode. The probability
         * @param stateIndex index of the state to add the exit point
         * @param proba probability to exit the gesture from this state
         * @throws runtime_error if the model is not hierarchical
         * @throws out_of_range if the state index is out of bounds
         */
        void addExitPoint(int stateIndex, float proba);
        
        ///@}
        
#pragma mark > Play!
        /** @name Playing */
        ///@{
        
        /**
         * @brief Initialize the 'Performance' phase
         */
        void performance_init();
        
        /**
         * @brief Main Play function: performs recognition (unimodal mode) or regression (bimodal mode)
         * @details The predicted output is stored in the observation vector in bimodal mode
         * @param observation pointer to current observation vector. Must be of size 'dimension' (input + output dimension).
         * @return likelihood computed on the gesture modality by a forward algorithm
         */
        double performance_update(std::vector<float> const& observation);
        
        ///@}
        
#pragma mark > JSON I/O
        /** @name JSON I/O */
        ///@{
        
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
        
        ///@}
        
#pragma mark > Conversion & Extraction
        /** @name Conversion & Extraction */
        ///@{
        
        
        /**
         * @brief Convert to bimodal HMM in place
         * @param dimension_input dimension of the input modality
         * @throws runtime_error if the model is already bimodal
         * @throws out_of_range if the requested input dimension is too large
         */
        void make_bimodal(unsigned int dimension_input);
        
        /**
         * @brief Convert to unimodal HMM in place
         * @throws runtime_error if the model is already unimodal
         */
        void make_unimodal();
        
        /**
         * @brief extract a submodel with the given columns
         * @param columns columns indices in the target order
         * @throws runtime_error if the model is training
         * @throws out_of_range if the number or indices of the requested columns exceeds the current dimension
         * @return a HMM from the current model considering only the target columns
         */
        HMM extract_submodel(std::vector<unsigned int>& columns) const;
        
        /**
         * @brief extract the submodel of the input modality
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a unimodal HMM of the input modality from the current bimodal model
         */
        HMM extract_submodel_input() const;
        
        /**
         * @brief extract the submodel of the output modality
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a unimodal HMM of the output modality from the current bimodal model
         */
        HMM extract_submodel_output() const;
        
        /**
         * @brief extract the model with reversed input and output modalities
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a bimodal HMM  that swaps the input and output modalities
         */
        HMM extract_inverse_model() const;
        
        ///@}
        
#pragma mark -
#pragma mark === Public Attributes ===
        /**
         * @brief Results: Estimated time progression.
         * @details The time progression is computed as the centroid of the state
         * probability distribution estimated by the forward algorithm
         */
        double results_progress;
        
        /**
         * @brief Results: Likelihood to exit the gesture on the next time step
         */
        double results_exit_likelihood;
        
        /**
         * @brief Results: Likelihood to exit the gesture on the next time step (normalized -/- total likelihood)
         */
        double results_exit_ratio;
        
        /**
         * @brief Results: Index of the likeliest state
         */
        unsigned int results_likeliest_state;
        
        /**
         * @brief State probabilities estimated by the forward algorithm.
         */
        std::vector<double> alpha;
        
        /**
         * @brief State probabilities estimated by the hierarchical forward algorithm.
         * @details the variable is only allocated/used in hierarchical mode (see 'HIERARCHICAL' flag)
         */
        std::vector<double> alpha_h[3];
        
        /**
         * @brief States of the model (Gaussian Mixture Models)
         */
        std::vector<GMM> states;
        
        /**
         * @brief Prior probabilities
         */
        std::vector<float> prior;
        
        /**
         * @brief Transition Matrix
         */
        std::vector<float> transition;
        
#ifndef XMM_TESTING
    protected:
#endif
        
        ///@cond DEVDOC
        
#pragma mark -
#pragma mark === Protected Methods ===
#pragma mark > Utilities
        /** @name Utilities (protected) */
        ///@{
        
        /**
         * @brief Copy between 2 MHMM models (called by copy constructor and assignment methods)
         * @param src Source Model
         * @param dst Destination Model
         */
        using ProbabilisticModel::_copy;
        virtual void _copy(HMM *dst, HMM const& src);
        
        ///@}
        
#pragma mark > Parameters initialization
        /** @name Parameters initialization */
        ///@{
        
        /**
         * @brief Allocate model parameters
         */
        void allocate();
        
        /**
         * @brief Evaluate the number of hidden states based on the length of the training examples
         * @todo Handle Variable number of states in HMM Class
         */
        void evaluateNbStates(int factor = 5);
        
        /**
         * @brief initialize model parameters to their default values
         */
        void initParametersToDefault();
        
        /**
         * @brief initialize the means of each state with all training phrases (single gaussian)
         */
        void initMeansWithAllPhrases();
        
        /**
         * @brief initialize the covariances of each state with all training phrases (single gaussian)
         */
        void initCovariances_fullyObserved();
        
        /**
         * @brief initialize the means and covariances of each state using GMM-EM on segments.
         */
        void initMeansCovariancesWithGMMEM();
        
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
        
        ///@}
        
#pragma mark > Forward-Backward algorithm
        /** @name Forward-Backward Algorithm */
        ///@{
        
        /**
         * @brief Initialization of the forward algorithm
         * @param observation observation vector at time t. If the model is bimodal, this vector
         * should be only the observation on the input modality.
         * @param observation_output observation on the output modality (only used if the model is bimodal).
         * If unspecified, the update is performed on the input modality only.
         * @return instantaneous likelihood
         */
        double forward_init(const float* observation, const float* observation_output=NULL);
        
        /**
         * @brief Update of the forward algorithm
         * @param observation observation vector at time t. If the model is bimodal, this vector
         * should be only the observation on the input modality.
         * @param observation_output observation on the output modality (only used if the model is bimodal).
         * If unspecified, the update is performed on the input modality only.
         * @return instantaneous likelihood
         */
        double forward_update(const float* observation, const float* observation_output=NULL);
        
        /**
         * @brief Initialization Backward algorithm
         * @param ct inverse of the likelihood at time step t computed
         * with the forward algorithm (see Rabiner 1989)
         */
        void backward_init(double ct);
        
        /**
         * @brief Update of the Backward algorithm
         * @param ct inverse of the likelihood at time step t computed
         * with the forward algorithm (see Rabiner 1989)
         * @param observation observation vector at time t. If the model is bimodal, this vector
         * should be only the observation on the input modality.
         * @param observation_output observation on the output modality (only used if the model is bimodal).
         * If unspecified, the update is performed on the input modality only.
         */
        void backward_update(double ct, const float* observation, const float* observation_output=NULL);
        
        ///@}
        
#pragma mark > Training
        /** @name Training (protected) */
        ///@{
        
        /**
         * @brief Initialization of the parameters before training
         */
        void train_EM_init();
        
        /**
         * @brief Termination of the training algorithm
         */
        void train_EM_terminate();
        
        /**
         * @brief update method of the EM algorithm (calls Baum-Welch Algorithm)
         */
        virtual double train_EM_update();
        
        /**
         * @brief Compute the forward-backward algorithm on a phrase of the training set
         * @param currentPhrase pointer to the phrase of the training set
         * @param phraseIndex index of the phrase
         * @return lieklihood of the phrase given the model's current parameters
         */
        double baumWelch_forwardBackward(Phrase* currentPhrase, int phraseIndex);
        
        /**
         * @brief Update of the forward algorithm for Training (observation probabilities are pre-computed)
         * @param observation_likelihoods likelihoods of the observations for each state
         * @return instantaneous likelihood
         */
        double baumWelch_forward_update(std::vector<double>::iterator observation_likelihoods);
        
        /**
         * @brief Update of the Backward algorithm for Training (observation probabilities are pre-computed)
         * @param ct inverse of the likelihood at time step t computed
         * with the forward algorithm (see Rabiner 1989)
         * @param observation_likelihoods likelihoods of the observations for each state
         */
        void baumWelch_backward_update(double ct, std::vector<double>::iterator observation_likelihoods);
        
        /**
         * @brief Compute the sum of the gamma variable (for use in EM)
         */
        void baumWelch_gammaSum();
        
        /**
         * @brief Estimate the Coefficients of the Gaussian Mixture for each state
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
        
        ///@}
        
#pragma mark > Play!
        /** @name Playing (protected) */
        ///@{
        
        /**
         * @brief Adds a cyclic Transition probability (from last state to first state)
         * @details avoids getting stuck at the end of the model. this method is idle for a hierarchical model.
         * @param proba probability of the transition form last to first state
         */
        void addCyclicTransition(double proba);
        
        /**
         * @brief Estimates the likeliest state and compute the bounds of the windows over the states.
         * @details The window is centered around the likeliest state, and its size is the number of states.
         * The window is clipped to the first and last states.
         */
        void updateAlphaWindow();
        
        /**
         * @brief Compute the regression for the case of a bimodal model, given the estimated
         * state probabilities estimated by forward algorithm
         * @param observation_input observation on the input modality
         * @param predicted_output output predicted by non-linear regression.
         */
        void regression(std::vector<float> const& observation_input, std::vector<float>& predicted_output);
        
        /**
         * @brief Updates the normalized time progression in the results.
         * @details The time progression is computed as the expected value of the states probabilities,
         * normalized between 0 and 1.
         */
        void updateTimeProgression();
        
        ///@}
        
#pragma mark > Exit Probabilities
        /** @name Exit Probabilities: update */
        ///@{
        
        /**
         * @brief Update the exit probability vector given the probabilities
         * @details this method is only active in 'HIERARCHICAL' mode. The probability
         * vector defines the probability of exiting the gesture from each state. If unspecified,
         * only the last state of the gesture has a non-zero probability.
         * @param _exitProbabilities vector of exit probabilities (size must be nbStates)
         * @throws runtime_error if the model is not hierarchical
         */
        void updateExitProbabilities(float *_exitProbabilities = NULL);
        
        ///@}
        
#pragma mark -
#pragma mark === Protected Attributes ===
        /**
         * @brief Number of hidden states of the Markov chain
         */
        int nbStates_;
        
        /**
         * @brief Number of Gaussian mixture components for each state
         */
        int nbMixtureComponents_;
        
        /**
         * @brief Offset added to the diagonal of covariance matrices for convergence (relative to data variance)
         */
        double  varianceOffset_relative_;
        
        /**
         * @brief Offset added to the diagonal of covariance matrices for convergence (minimum value)
         */
        double  varianceOffset_absolute_;
        
        /**
         * @brief Covariance Mode
         */
    public:
        GaussianDistribution::COVARIANCE_MODE covariance_mode_;
    protected:
        /**
         * @brief Transition mode of the model (left-right vs ergodic)
         */
        TRANSITION_MODE transitionMode_;
        
        /**
         * @brief Defines if the forward algorithm has been initialized
         */
        bool forwardInitialized_;
        
        /**
         * @brief Define if the EM algorithm should re-estimate the means of each state.
         */
        bool estimateMeans_;
        
        /**
         * @brief used to store the alpha estimated at the previous time step
         */
        std::vector<double> previousAlpha_;
        
        /**
         * @brief backward state probabilities
         */
        std::vector<double> beta_;
        
        /**
         * @brief used to store the beta estimated at the previous time step
         */
        std::vector<double> previousBeta_;
        
        /**
         * @brief Sequence of Gamma probabilities
         */
        std::vector< std::vector<double> > gammaSequence_;
        
        /**
         * @brief Sequence of Epsilon probabilities
         */
        std::vector< std::vector<double> > epsilonSequence_;
        
        /**
         * @brief Sequence of Gamma probabilities for each mixture component
         */
        std::vector< std::vector< std::vector<double> > > gammaSequencePerMixture_;
        
        /**
         * @brief Sequence of alpha (forward) probabilities
         */
        std::vector<double> alpha_seq_;
        
        /**
         * @brief Sequence of beta (backward) probabilities
         */
        std::vector<double> beta_seq_;
        
        /**
         * @brief Used to store the sums of the gamma variable
         */
        std::vector<double> gammaSum_;
        
        /**
         * @brief Used to store the sums of the gamma variable for each mixture component
         */
        std::vector<double> gammaSumPerMixture_;
        
        /**
         * @brief Defines if the model is a submodel of a hierarchical HMM.
         * @details in practice this adds exit probabilities to each state. These probabilities are set
         * to the last state by default.
         */
        bool is_hierarchical_;
        
        /**
         * @brief Exit probabilities for a hierarchical model.
         */
        std::vector<float> exitProbabilities_;
        
        /**
         * @brief Type of regression estimator (default = FULL)
         * @see REGRESSION_ESTIMATOR
         */
        REGRESSION_ESTIMATOR regression_estimator_;
        
        /**
         * @brief minimum index of the alpha window (used for regression & time progression)
         */
        int results_window_minindex;
        
        /**
         * @brief minimum index of the alpha window (used for regression & time progression)
         */
        int results_window_maxindex;
        
        /**
         * @brief normalization constant of the alpha window (used for regression & time progression)
         */
        double results_window_normalization_constant;
        
        ///@endcond
    };
}

#endif
