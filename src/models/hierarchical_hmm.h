/*
 * hierarchical_hmm.h
 *
 * Hierarchical Hidden Markov Model for continuous recognition and regression
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

#ifndef xmm_lib_hierarchical_hmm_h
#define xmm_lib_hierarchical_hmm_h

#include "model_group.h"
#include "hmm.h"

namespace xmm
{
    /**
     * @ingroup HMM
     * @class HierarchicalHMM
     * @brief Hierarchical Hidden Markov Model
     * @todo Needs more details
     */
    class HierarchicalHMM : public ModelGroup< HMM > {
    public:
#pragma mark -
#pragma mark === Public Interface ===
        /**
         * @brief Default exit transition for the highest level
         */
        static const double DEFAULT_EXITTRANSITION() { return 0.1; }
        
        /**
         * @brief Default value for the Incremental Learning update of the transition matrix
         */
        static const bool DEFAULT_INCREMENTALLEARNING = false;
        
        /**
         * @brief Default regularization factor for the update of transition
         * probabilities for incremental learning
         */
        static const bool DEFAULT_REGULARIZATIONFACTOR = false;
        
        /**
         * @brief Iterator over models
         */
        typedef std::map<Label, HMM>::iterator model_iterator;
        
        /**
         * @brief Constant Iterator over models
         */
        typedef std::map<Label, HMM>::const_iterator const_model_iterator;
        
#pragma mark > Constructor
        /*@{*/
        /** @name Constructor */
        /**
         * @brief Constructor
         * @param flags Construction flags
         * @param _globalTrainingSet Global training set
         * @param covariance_mode Covariance Mode (Full vs diagonal)
         */
        HierarchicalHMM(xmm_flags flags = NONE,
                        TrainingSet *_globalTrainingSet = NULL,
                        GaussianDistribution::COVARIANCE_MODE covariance_mode = GaussianDistribution::FULL);
        
        
        /**
         * @brief Copy Constructor
         * @param src Source Model
         */
        HierarchicalHMM(HierarchicalHMM const& src);
        
        /**
         * @brief Assignment
         * @param src Source Model
         */
        HierarchicalHMM& operator=(HierarchicalHMM const& src);
        
        /**
         * @brief Destructor
         */
        virtual ~HierarchicalHMM();
        
        /**
         * @brief Remove All models
         */
        virtual void clear();
        
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
         * @brief Get Offset added to covariance matrices for convergence (Relative to data variance)
         * @return Offset added to covariance matrices for convergence
         */
        double get_varianceOffset_relative() const;
        
        /**
         * @brief Get Offset added to covariance matrices for convergence (Minimum value)
         * @return Offset added to covariance matrices for convergence
         */
        double get_varianceOffset_absolute() const;
        
        /**
         * @brief Set the offset to add to the covariance matrices
         * @param varianceOffset_relative offset to add to the diagonal of covariance matrices (relative to data variance)
         * @param varianceOffset_absolute offset to add to the diagonal of covariance matrices (minimum value)
         * @throws invalid_argument if the covariance offset is <= 0
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
        HMM::REGRESSION_ESTIMATOR get_regression_estimator() const;
        
        /**
         * @brief Set the regression estimator type
         * @param regression_estimator type of estimator
         * @see REGRESSION_ESTIMATOR
         */
        void set_regression_estimator(HMM::REGRESSION_ESTIMATOR regression_estimator);
        
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
        
        bool get_estimateMeans() const;
        
        void set_estimateMeans(bool _estimateMeans);
        
        void addExitPoint(int state, float proba);
        
        /**
         * @brief return learning mode: "incremental" or "ergodic"
         * @details  if ergodic, each time a model is added at the high level, the transition
         * matrix is reset to ergodic. if "incremental", the transitions are updated using regularization
         * @return learningMode "incremental" or "ergodic"
         */
        std::string get_learningMode() const;
        
        /**
         * @brief set learning mode: "incremental" or "ergodic"
         * @details  if ergodic, each time a model is added at the high level, the transition
         * matrix is reset to ergodic. if "incremental", the transitions are updated using regularization
         * @param learningMode "incremental" or "ergodic"
         * @throws invalid_argument if the argument is neither "incremental" nor "ergodic"
         */
        void set_learningMode(std::string learningMode);
        
        /**
         * @brief get a copy of the high-level Prior probabilities vector
         * @param prior output High-level prior probability vector
         */
        void get_prior(std::vector<double>& prior) const;
        
        /**
         * @brief set high-level prior probabilities vector
         * @param prior high-level probability vector (size nbPrimitives)
         * @throws invalid_argument if the array has a wrong format (not enough values)
         * @warning the models are ordered in ascending order by label
         */
        void set_prior(std::vector<double> const& prior);
        
        /**
         * @brief get a copy of the high-level transition matrix
         * @param trans output high-level transition matrix
         */
        void get_transition(std::vector<double>& trans) const;
        
        /**
         * @brief set the high-level  transition matrix
         * @param trans high-level transition matrix
         * @throws invalid_argument if the array has a wrong format (not enough values)
         * @warning the models are ordered in ascending order by label
         */
        void set_transition(std::vector<double> const& trans);
        
        /**
         * @brief get a copy of the high-level exit probabilities
         * @details exit probabilities are the probabilities to finish and go
         * back to the root
         * @param trans output exit transition vector of the high level
         */
        void get_exitTransition(std::vector<double>& trans) const;
        
        /**
         * @brief set the exit transition vector of the high level
         * @param exittrans high-level exit probabilities vector
         * @throws invalid_argument if the array has a wrong format (not enough values)
         * @warning the models are ordered in ascending order by label
         */
        void set_exitTransition(std::vector<double> const& exittrans);
        
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
        virtual void performance_update(std::vector<float> const& observation);
        
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
        
#pragma mark > Conversion & Extraction
        /*@{*/
        /** @name Conversion & Extraction */
        
        /**
         * @brief Convert to bimodal HierarchicalHMM in place
         * @param dimension_input dimension of the input modality
         * @throws runtime_error if the model is already bimodal
         * @throws out_of_range if the requested input dimension is too large
         */
        void make_bimodal(unsigned int dimension_input);
        
        /**
         * @brief Convert to unimodal HierarchicalHMM in place
         * @throws runtime_error if the model is already unimodal
         */
        void make_unimodal();
        
        /**
         * @brief extract a submodel with the given columns
         * @param columns columns indices in the target order
         * @throws runtime_error if the model is training
         * @throws out_of_range if the number or indices of the requested columns exceeds the current dimension
         * @return a HierarchicalHMM from the current model considering only the target columns
         */
        HierarchicalHMM extract_submodel(std::vector<unsigned int>& columns) const;
        
        /**
         * @brief extract the submodel of the input modality
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a unimodal GMMGroup of the input modality from the current bimodal model
         */
        HierarchicalHMM extract_submodel_input() const;
        
        /**
         * @brief extract the submodel of the output modality
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a unimodal HierarchicalHMM of the output modality from the current bimodal model
         */
        HierarchicalHMM extract_submodel_output() const;
        
        /**
         * @brief extract the model with reversed input and output modalities
         * @throws runtime_error if the model is training or if it is not bimodal
         * @return a bimodal HierarchicalHMM  that swaps the input and output modalities
         */
        HierarchicalHMM extract_inverse_model() const;
        
        /*@}*/
        
#pragma mark -
#pragma mark === Public attributes ===
        /**
         * @brief Prior probabilities of the models
         */
        std::map<Label, double> prior;
        
        /**
         * @brief exit probabilities of the model (probability to finish and go back to the root)
         */
        std::map<Label, double> exitTransition;
        
        /**
         * @brief Transition probabilities between models
         */
        std::map<Label, std::map<Label, double> > transition;
        
    protected:
#pragma mark -
#pragma mark === Protected Methods ===
        /**
         * @brief Copy between two Hierarhical HMMs
         * @param src Source Model
         * @param dst Destination Model
         */
        virtual void _copy(HierarchicalHMM *dst, HierarchicalHMM const& src);
        
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
         * @brief Initialization of the Forward Algorithm for the hierarchical HMM.
         * see: Jules Françoise. Realtime Segmentation and Recognition of Gestures using Hierarchical Markov Models. Master’s Thesis, Université Pierre et Marie Curie, Ircam, 2011. [http://articles.ircam.fr/textes/Francoise11a/index.pdf]
         * @param observation observation vector. If the model is bimodal, this should be allocated for
         * both modalities, and should contain the observation on the input modality. The predicted
         * output will be appended to the input modality observation
         */
        void forward_init(std::vector<float> const& observation);
        
        /**
         * @brief Update of the Forward Algorithm for the hierarchical HMM.
         * see: Jules Françoise. Realtime Segmentation and Recognition of Gestures using Hierarchical Markov Models. Master’s Thesis, Université Pierre et Marie Curie, Ircam, 2011. [http://articles.ircam.fr/textes/Francoise11a/index.pdf]
         * @param observation observation vector. If the model is bimodal, this should be allocated for
         * both modalities, and should contain the observation on the input modality. The predicted
         * output will be appended to the input modality observation
         * @todo check if the algorithm is right for an ergodic transition structure of the sub-hmms
         */
        void forward_update(std::vector<float> const& observation);
        
        /**
         * @brief get instantaneous likelihood
         *
         * get instantaneous likelihood on alpha variable for exit state exitNum.
         * @param exitNum number of exit state (0=continue, 1=transition, 2=back to root). if -1, get likelihood over all exit states
         * @param likelihoodVector likelihood vector (size nbPrimitives)
         */
        void likelihoodAlpha(int exitNum, std::vector<double> &likelihoodVector) const;
        
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
        std::vector<double> V1_;
        
        /**
         * @brief intermediate Forward variable (used in Frontier algorithm)
         */
        std::vector<double> V2_;
    };
    
}

#endif