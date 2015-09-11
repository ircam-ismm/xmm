/*
 * probabilistic_model.h
 *
 * Abstract class for Probabilistic Machine learning models based on the EM algorithm
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

#ifndef xmm_lib_probabilistic_model_h
#define xmm_lib_probabilistic_model_h

#include "ringbuffer.h"
#include "training_set.h"

#ifdef USE_PTHREAD
#include <pthread.h>
#endif

namespace xmm
{
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
    
    ///@internal
    
    /**
     * @ingroup Core
     * @class ProbabilisticModel
     * @brief Generic Template for Machine Learning Probabilistic models based on the EM algorithm
     */
    class ProbabilisticModel : public Listener, public Writable
    {
    public:
        template<typename modelType> friend class ModelGroup;
        friend class GMMGroup;
        friend class HierarchicalHMM;
        
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
            TRAINING_ERROR,
            
            /**
             * The training has been cancelled.
             */
            TRAINING_ABORT,
            
            /**
             * The training of all classes has finished
             */
            TRAINING_ALLDONE
        };
        
        ///@cond DEVDOC
        
        /**
         * Default value for the minimum number of steps of the EM algorithm
         */
        static const int DEFAULT_EMSTOP_MINSTEPS = 10;
        
        /**
         * Default value for the maximum number of steps of the EM algorithm
         */
        static const int DEFAULT_EMSTOP_MAXSTEPS = 0;
        
        /**
         * Default value for the percent-change criterion of the EM algorithm
         */
        static const double DEFAULT_EMSTOP_PERCENT_CHG() { return 0.01; }
        
        /**
         * Default value for the size of the likelihood window used for smoothing likelihoods
         * in performance mode.
         */
        static const int DEFAULT_LIKELIHOOD_WINDOW = 1;
        
        /**
         * Default absolute maximum number of EM iterations
         */
        static const int DEFAULT_EMSTOP_ABSOLUTEMAXSTEPS = 100;
        
        ///@endcond
        
#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Constructors
        /** @name Constructors */
        ///@{
        
        /**
         * @brief Constructor
         * @param trainingSet training set associated with the model
         * @param flags Construction Flags. The only valid flag here is BIMODAL, that defines if the
         * model is bimodal (can be used for regression).
         */
        ProbabilisticModel(xmm_flags flags = NONE, TrainingSet *trainingSet = NULL);
        
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
        ///@}
        
#pragma mark > Accessors
        /** @name Accessors */
        ///@{
        
        /**
         * @brief Checks if the model is training
         * @return true if the model is training
         */
        bool is_training() const;
        
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
        unsigned int dimension() const;
        
        /**
         * @brief Get the dimension of the input modality
         * @warning This can only be used in bimodal mode (construction with 'BIMODAL' flag)
         * @return dimension of the input modality
         * @throws runtime_error if not in bimodal mode
         */
        unsigned int dimension_input() const;
        
        /**
         * @brief get size of the likelihood smoothing buffer (number of frames)
         * @return size of the likelihood smoothing buffer
         */
        unsigned int get_likelihoodwindow() const;
        
        /**
         * @brief set size of the likelihood smoothing buffer (number of frames)
         * @param likelihoodwindow size of the likelihood smoothing buffer
         * @throws invalid_argument if likelihoodwindow is < 1
         */
        void set_likelihoodwindow(unsigned int likelihoodwindow);
        
        /**
         * @brief get a copy of the column names of the input/output data
         */
        std::vector<std::string> const& get_column_names() const;
        
        ///@}
        
#pragma mark > Training
        /** @name Training */
        ///@{
        
        /**
         * @brief Main training method based on the EM algorithm
         * @details the method performs a loop over the pure virtual method train_EM_update() until convergence.
         * The @a train_EM_update method computes both E and M steps of the EM algorithm.
         * @see train_EM_update
         */
        void train();
        
#ifdef USE_PTHREAD
        /**
         * @brief Interrupt the current training function
         * @param this_thread pointer to the training thread
         * @return true if the model was training and has been requested to cancel
         * @warning only defined if USE_PTHREAD is defined
         */
        bool abortTraining(pthread_t this_thread);
#endif
        
        /**
         * @brief set the callback function associated with the training algorithm
         * @details the function is called whenever the training is over or an error happened during training
         */
        void set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata);
        
        ///@}
        
        /**
         * @brief Function pointer for parallel training
         * @param context pointer to the object to train
         */
        static void* train_func(void *context);
        
#pragma mark > Performance
        /** @name Performance */
        ///@{
        
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
        virtual double performance_update(std::vector<float> const& observation) = 0;
        
        ///@}
        
#pragma mark > JSON I/O
        /** @name JSON I/O */
        ///@{
        
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
        
        ///@}
        
        
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
         * @brief Stop criterion of the EM algorithm
         * @see EMStopCriterion
         */
        EMStopCriterion stopcriterion;
        
        /**
         * progression within the training algorithm
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
        std::vector<float> results_predicted_output;
        
        /**
         * @brief Conditional Output Variance
         * @warning this variable is not allocated if the model is not bimodal
         */
        std::vector<double> results_output_variance;
        
        /**
         * Log-likelihood of the data given the model's parameters at the en of training
         */
        double trainingLogLikelihood;
        
        /**
         * Number of EM iterations
         */
        double trainingNbIterations;
        
#ifndef XMM_TESTING
    protected:
#endif
        ///@cond DEVDOC
        
#pragma mark -
#pragma mark === Protected Methods ===
        /** @name Utilities (protected) */
        ///@{
        
        /**
         * @brief Copy between two models
         * @param src Source Model
         * @param dst Destination Model
         */
        virtual void _copy(ProbabilisticModel *dst, ProbabilisticModel const& src);
        
        /**
         * @brief Prevents the attribute for being changed during training.
         * @throws runtime_error if the model is training.
         */
        inline void prevent_attribute_change() const
        {
#ifdef USE_PTHREAD
            if (this->is_training())
                throw std::runtime_error("Cannot set attributes during Training");
#endif
        }
        
        /**
         * @brief Checks if the model is still training
         * @throws runtime_error if the model is training.
         */
        inline void check_training() const
        {
#ifdef USE_PTHREAD
            if (this->is_training())
                throw std::runtime_error("The model is training");
#endif
        }
        
        /**
         * @brief update the content of the likelihood buffer and return average likelihood.
         * @details The method also updates the cumulative log-likelihood computed over a window (cumulativeloglikelihood)
         * @param instantLikelihood instantaneous likelihood at the current step
         */
        void updateLikelihoodBuffer(double instantLikelihood);
        
        /**
         * @brief handle notifications of the training set
         * @details here only the dimensions attributes of the training set are considered
         * @param attribute name of the attribute: should be either "dimension" or "dimension_input"
         */
        void notify(std::string attribute);
        
        /**
         * @brief Allocate memory for the model's parameters
         * @details called when dimensions are modified
         */
        virtual void allocate() = 0;
        
        ///@}
        
        /** @name Training (protected) */
        ///@{
        
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
        bool train_EM_hasConverged(int step, double log_prob, double old_log_prob) const;
        
        /**
         * @brief checks if a cancel request has been sent and accordingly cancels the training process
         * @return true if the training has been canceled.
         */
        bool check_and_cancel_training();
        
        ///@}
        
#pragma mark -
#pragma mark === Protected Attributes ===
        /**
         * @brief Construction flags
         */
        xmm_flags flags_;
        
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
        void (*trainingCallbackFunction_)(void *srcModel, CALLBACK_FLAG state, void* extradata);
        
        /**
         * @brief Extra data to pass in argument to the callback function
         */
        void *trainingExtradata_;
        
        /**
         * @brief Likelihood buffer used for smoothing
         */
        RingBuffer<double, 1> likelihoodBuffer_;
        
        /**
         * @brief labels of the columns of input/output data (e.g. descriptor names)
         */
        std::vector<std::string> column_names_;
        
#ifdef USE_PTHREAD
        /**
         * @brief Mutex used in Concurrent Mode
         * @warning only defined if USE_PTHREAD is defined
         */
        pthread_mutex_t trainingMutex;
        
        /**
         * @brief defines if the model is being trained.
         */
        bool is_training_;
        
        /**
         * @brief defines if the model received a request to cancel training
         */
        bool cancel_training_;
#endif
        ///@endcond
    };
    
    ///@endinternal
}

#endif
