//
// concurrent_models.h
//
// Multiple machine learning models running in parallel
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef rtml_model_group_h
#define rtml_model_group_h

#include "probabilistic_model.h"

#ifdef USE_PTHREAD
#include <pthread.h>
#endif

using namespace std;

#pragma mark -
#pragma mark Class Definition
/**
 * @ingroup Base
 * @class ModelGroup
 * @brief Handle machine learning models running in parallel
 * @tparam ModelType type of the models (implemented: GMM, HMM)
 */
template<typename ModelType>
class ModelGroup : public Listener, public Writable
{
public:
#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Iterators
    /**
     * @enum GROUP_ESTIMATION_MODE
     * @brief Type of performance mode for concurrent models
     */
    enum GROUP_ESTIMATION_MODE {
        /**
         * @brief the performance_update method returns the results of the likeliest model
         */
        LIKELIEST = 0,
        
        /**
         * @brief the performance_update method returns a weighted sum of the results of each model
         */
        MIXTURE = 1
    };
    
    /**
     * @brief Iterator over models
     */
    typedef typename  map<Label, ModelType>::iterator model_iterator;
    
    /**
     * @brief Constant Iterator over models
     */
    typedef typename  map<Label, ModelType>::const_iterator const_model_iterator;
    
    /**
     * @brief Iterator over labels
     */
    typedef typename  map<int, Label>::iterator labels_iterator;
    
#pragma mark > Constructors
    /*@{*/
    /** @name Constructors */
    /**
     * @brief Constructor
     * @param globalTrainingSet global training set: contains all phrases for each model
     * @param flags Construction flags: use 'BIMODAL' for a use with Regression / Generation.
     * For the Hierarchial HMM, use 'HIERARCHICAL' to specify the submodels they are embedded
     * in a hierarchical structure
     */
    ModelGroup(rtml_flags flags = NONE,
               TrainingSet *globalTrainingSet=NULL)
    : trainingCallbackFunction_(NULL)
    {
        bimodal_ = (flags & BIMODAL);
        this->globalTrainingSet = globalTrainingSet;
        if (this->globalTrainingSet)
            this->globalTrainingSet->add_listener(this);
        referenceModel_ = ModelType(flags, this->globalTrainingSet);
        referenceModel_.set_trainingCallback(monitor_training, this);
        performanceMode_ = LIKELIEST;
        models_to_train_ = 0;
    }
    
    /**
     * @brief Destructor
     */
    virtual ~ModelGroup()
    {
        clear();
        if (this->globalTrainingSet)
            this->globalTrainingSet->remove_listener(this);
    }
    
    /*@}*/
    
#pragma mark Tests & Utilies
    /*@{*/
    /** @name Tests & Utilies */
    /**
     * @brief Check if at least 1 model is still training
     * @return true if at least 1 model is still training
     */
    bool is_training() const {
        return (models_to_train_ > 0);
    }
    
    /**
     * @brief Check if a model has been trained
     * @param label class label of the model
     * @return true if the model has been trained and the training data has not been
     * modified in between
     * @throw out_of_range if the label does not exist
     */
    bool is_trained(Label const& label) const
    {
        if (models.find(label) == models.end())
            throw out_of_range("Class " + label.as_string() + " does not exist");
        return models.at(label).trained;
    }
    
    /**
     * @brief Check if all models have been trained
     * @return true if all the models has been trained and the training data has not been
     * modified in between
     */
    bool is_trained() const
    {
        if (size() == 0)
            return false;
        for (const_model_iterator it = models.begin(); it != models.end(); ++it) {
            if (!it->second.trained)
                return false;
        }
        return true;
    }
    
    /**
     * @brief get the number of models
     * @return number of models
     */
    unsigned int size() const
    {
        return (unsigned int)(models.size());
    }
    
    /**
     * @brief Remove All models
     */
    virtual void clear()
    {
#ifdef USE_PTHREAD
        if (is_training())
            stopTraining();
#endif
        models.clear();
    }
    
    /**
     * @brief Remove Specific model
     * @param label label of the model
     * @throw out_of_range if the label does not exist
     */
    virtual void remove(Label const& label)
    {
        model_iterator it = models.find(label);
        if (it == models.end())
            throw out_of_range("Class " + label.as_string() + " does not exist");
        models.erase(it);
    }
    
    
    /*@}*/
    
#pragma mark > Accessors
    /*@{*/
    /** @name Accessors */
    /**
     * @brief Set pointer to the global training set
     * @param globalTrainingSet pointer to the global training set
     */
    void set_trainingSet(TrainingSet *globalTrainingSet)
    {
        if (this->globalTrainingSet)
            this->globalTrainingSet->remove_listener(this);
        this->globalTrainingSet = globalTrainingSet;
        if (this->globalTrainingSet)
            this->globalTrainingSet->add_listener(this);
        referenceModel_.set_trainingSet(this->globalTrainingSet);
    }
    
    /**
     * @brief Get Total Dimension of the model (sum of dimension of modalities)
     * @return total dimension of Gaussian Distributions
     */
    int dimension() const
    {
        return this->referenceModel_.dimension();
    }
    
    /**
     * @brief Get the dimension of the input modality
     * @warning This can only be used in bimodal mode (construction with 'BIMODAL' flag)
     * @return dimension of the input modality
     * @throws runtime_error if not in bimodal mode
     */
    int dimension_input() const
    {
        if (!bimodal_)
            throw runtime_error("Model is not bimodal");
        return this->referenceModel_.dimension_input();
    }
    
    /**
     * @brief Sets the performance mode (likeliest vs mixture)
     * @param performanceMode_str performance mode: if "likeliest", the performance_update function estimates
     * the output modality with the likeliest model. If "mixture",  the performance_update function estimates
     * the output modality as a weighted sum of all models' predictions.
     * @throws invalid_argument if the argument is not "likeliest" or "mixture"
     */
    void set_performanceMode(string performanceMode_str)
    {
        if (!performanceMode_str.compare("likeliest")) {
            performanceMode_ = LIKELIEST;
        } else if (!performanceMode_str.compare("mixture")) {
            performanceMode_ = MIXTURE;
        } else {
            throw invalid_argument("Unknown performance mode for multiple models");
        }
    }
    
    /**
     * @brief Get the performance mode (likeliest vs mixture)
     * @return performance mode: if "likeliest", the performance_update function estimates
     * the output modality with the likeliest model. If "mixture",  the performance_update function estimates
     * the output modality as a weighted sum of all models' predictions.
     */
    string get_performanceMode()
    {
        if (performanceMode_ == LIKELIEST)
            return "likeliest";
        else
            return "mixture";
    }
    
    /**
     * @brief Get minimum number of EM steps
     * @return minimum number of steps of the EM algorithm
     */
    unsigned int get_EM_minSteps() const
    {
        return this->referenceModel_.stopcriterion.minSteps;
    }
    
    /**
     * @brief Get maximum number of EM steps
     * @return maximum number of steps of the EM algorithm
     * @see EMStopCriterion
     */
    unsigned int get_EM_maxSteps() const
    {
        return this->referenceModel_.stopcriterion.maxSteps;
    }
    
    /**
     * @brief Get EM convergence threshold in percent-change of the likelihood
     * @return loglikelihood percent-change convergence threshold
     * @see EMStopCriterion
     */
    double get_EM_percentChange() const
    {
        return this->referenceModel_.stopcriterion.percentChg;
    }
    
    /**
     * @brief Set minimum number of steps of the EM algorithm
     * @param steps minimum number of steps of the EM algorithm
     * @throws invalid_argument if steps < 1
     */
    void set_EM_minSteps(unsigned int steps)
    {
        this->referenceModel_.stopcriterion.minSteps = steps;
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.stopcriterion.minSteps = steps;
        }
    }
    
    /**
     * @brief Set maximum number of steps of the EM algorithm
     * @param steps maximum number of steps of the EM algorithm
     * @throws invalid_argument if steps < 1
     */
    void set_EM_maxSteps(unsigned int steps)
    {
        this->referenceModel_.stopcriterion.maxSteps = steps;
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.stopcriterion.maxSteps = steps;
        }
    }
    
    /**
     * @brief Set convergence threshold in percent-change of the likelihood
     * @param logLikPercentChg_ log-likelihood percent-change convergence threshold
     * @throws invalid_argument if logLikelihoodPercentChg <= 0
     */
    void set_EM_percentChange(double logLikPercentChg_)
    {
        this->referenceModel_.stopcriterion.percentChg = logLikPercentChg_;
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.stopcriterion.percentChg = logLikPercentChg_;
        }
    }
    
    /**
     * @brief get size of the likelihood smoothing buffer (number of frames)
     * @return size of the likelihood smoothing buffer
     */
    
    unsigned int get_likelihoodwindow() const
    {
        return this->referenceModel_.get_likelihoodwindow();
    }
    
    /**
     * @brief set size of the likelihood smoothing buffer (number of frames)
     * @param likelihoodwindow size of the likelihood smoothing buffer
     * @throws invalid_argument if likelihoodwindow is < 1
     */
    
    void set_likelihoodwindow(unsigned int likelihoodwindow)
    {
        this->referenceModel_.set_likelihoodwindow(likelihoodwindow);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_likelihoodwindow(likelihoodwindow);
        }
    }
    
    /*@}*/
    
#pragma mark > Training
    /*@{*/
    /** @name Training */
    /**
     * @brief Train a specific model
     * @details  The model is trained even if the dataset has not changed
     * @param label label of the model
     * @throw out_of_range if the label does not exist
     */
    virtual void train(Label const& label)
    {
#ifdef USE_PTHREAD
        stopTraining(label);
#endif
        updateTrainingSet(label);
        
#ifdef USE_PTHREAD
        pthread_create(&(training_threads[label]), NULL, &ModelType::train_func, &(this->models[label]));
        models_to_train_++;
        if (!trainingCallbackFunction_) {
            pthread_join(training_threads[label], NULL);
        }
#else
        models[label].train();
#endif
    }
    
    /**
     * @brief Train All model even if their data have not changed.
     */
    virtual void train()
    {
#ifdef USE_PTHREAD
        stopTraining();
#endif
        
        updateAllTrainingSets();
        
#ifdef USE_PTHREAD
        for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
            pthread_create(&(training_threads[it->first]), NULL, &ModelType::train_func, &(it->second));
            models_to_train_++;
        }
        if (!trainingCallbackFunction_) {
            for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
                pthread_join(training_threads[it->first], NULL);
            }
        }
#else
        // Sequential training
        for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
            it->second.train();
        }
#endif
    }
    
    /**
     * @brief Train all model which data has changed.
     */
    virtual void retrain()
    {
#ifdef USE_PTHREAD
        stopTraining();
#endif
        updateAllTrainingSets();
        
#ifdef USE_PTHREAD
        for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
            if (!it->second.trained || it->second.trainingSet->has_changed()) {
                pthread_create(&(training_threads[it->first]), NULL, &ModelType::train_func, &(it->second));
                if (trainingCallbackFunction_)
                    models_to_train_++;
            }
        }
        if (!trainingCallbackFunction_) {
            for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
                if (!it->second.trained || it->second.trainingSet->has_changed()) {
                    pthread_join(training_threads[it->first], NULL);
                }
            }
        }
#else
        // Sequential training
        for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
            if (!it->second.trained || it->second.trainingSet->has_changed()) {
                it->second.train();
            }
        }
#endif
    }
    
#ifdef USE_PTHREAD
    /**
     * @brief Aborts the training of a model
     * @warning only defined if USE_PTHREAD is defined
     * @param label label of the model to abort
     */
    void stopTraining(Label const& label) {
        if (this->models.find(label) != this->models.end())
            models[label].abortTraining(training_threads[label]);
    }
    
    /**
     * @brief Aborts training of all models
     * @warning only defined if USE_PTHREAD is defined
     */
    void stopTraining() {
        for (set<Label>::iterator label_it = globalTrainingSet->allLabels.begin(); label_it != globalTrainingSet->allLabels.end(); label_it++) {
            stopTraining(*label_it);
        }
    }
#endif
    
    /**
     * @brief Monitors the training of each model of the group.
     */
    static void monitor_training(void *model, CALLBACK_FLAG state, void *extradata)
    {
        ModelGroup<ModelType> *thismodelgroup = (ModelGroup<ModelType> *)extradata;
        if (state != TRAINING_RUN) {
            Label label = ((ModelType *)model)->trainingSet->getPhraseLabel(0);
            thismodelgroup->models_to_train_--;
#ifdef USE_PTHREAD
            thismodelgroup->training_threads.erase(label);
#endif
            if (state == TRAINING_ERROR || state == TRAINING_ABORT) {
                thismodelgroup->remove(label);
            }
        }
        if (thismodelgroup->trainingCallbackFunction_)
            thismodelgroup->trainingCallbackFunction_(model, state, thismodelgroup->trainingExtradata_);
    }
    
    /**
     * @brief set the callback function associated with the training algorithm
     * @details the function is called whenever the training is over or an error happened during training
     */
    void set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata) {
        this->referenceModel_.set_trainingCallback(monitor_training, this);
        for (model_iterator it=models.begin(); it != models.end(); ++it) {
            it->second.set_trainingCallback(monitor_training, this);
        }
        trainingExtradata_ = extradata;
        trainingCallbackFunction_ = callback;
    }
    
    /*@}*/
    
#pragma mark > Performance
    /*@{*/
    /** @name Performance */
    /**
     * @brief Initialize Performance
     */
    virtual void performance_init()
    {
        for (model_iterator it=this->models.begin(); it != this->models.end(); ++it) {
            it->second.performance_init();
        }
        results_instant_likelihoods.resize(size());
        results_normalized_instant_likelihoods.resize(size());
        results_normalized_likelihoods.resize(size());
        results_log_likelihoods.resize(size());
        if (bimodal_)
            results_predicted_output.resize(dimension() - dimension_input());
    }
    
    /**
     * @brief Update the results (Likelihoods)
     */
    virtual void update_likelihood_results()
    {
        double maxLogLikelihood;
        double normconst_instant(0.0);
        double normconst_smoothed(0.0);
        int i(0);
        for (model_iterator it=this->models.begin(); it != this->models.end(); ++it, ++i) {
            results_instant_likelihoods[i] = it->second.results_instant_likelihood;
            results_log_likelihoods[i] = it->second.results_log_likelihood;
            results_normalized_likelihoods[i] = exp(results_log_likelihoods[i]);
            
            normconst_instant += results_instant_likelihoods[i];
            normconst_smoothed += results_normalized_likelihoods[i];
            
            if (i == 0 || results_log_likelihoods[i] > maxLogLikelihood) {
                maxLogLikelihood = results_log_likelihoods[i];
                results_likeliest = it->first;
            }
        }
        
        i = 0;
        for (model_iterator it=this->models.begin(); it != this->models.end(); ++it, ++i) {
            results_normalized_likelihoods[i] /= normconst_smoothed;
            results_normalized_instant_likelihoods[i] = results_instant_likelihoods[i] / normconst_instant;
        }
    }
    
    /*@}*/
    
#pragma mark -
#pragma mark === Public attributes ===
    /**
     * @brief Models stored in a map. Each model is associated with a label
     */
    map<Label, ModelType> models;
    
    /**
     * @brief Global Training set for all labels
     */
    TrainingSet *globalTrainingSet;
    
    /**
     * @brief Result: Likelihood of each model
     */
    vector<double> results_instant_likelihoods;
    
    /**
     * @brief Result: Normalized Instantaneous Likelihood of each model
     */
    vector<double> results_normalized_instant_likelihoods;
    
    /**
     * @brief Result: Normalized Likelihood of each model
     */
    vector<double> results_normalized_likelihoods;
    
    /**
     * @brief Result: Windowed Cumulative Log-Likelihood of each model
     */
    vector<double> results_log_likelihoods;
    
    /**
     * @brief Result: Label of the likeliest model
     */
    Label results_likeliest;
    
    /**
     * Result: Predicted output modality observation
     */
    vector<float> results_predicted_output;
    
protected:
#pragma mark -
#pragma mark === Protected Methods ===
#pragma mark > Training Set
    /**
     * @brief Receives notifications from the global training set and dispatches to models
     */
    virtual void notify(string attribute)
    {
        referenceModel_.notify(attribute);
        for (model_iterator it=models.begin(); it != models.end(); ++it) {
            it->second.notify(attribute);
        }
        if (attribute == "destruction") {
            globalTrainingSet = NULL;
            return;
        }
    }
    
    /**
     * @brief Remove models which label is not in the training set anymore
     */
    virtual void removeDeprecatedModels()
    {
        globalTrainingSet->updateSubTrainingSets();
        
        // Look for deleted classes
        bool contLoop(true);
        while (contLoop) {
            contLoop = false;
            for (model_iterator it = models.begin(); it != models.end(); ++it) {
                if (globalTrainingSet->allLabels.find(it->first) == globalTrainingSet->allLabels.end())
                {
                    models.erase(it->first);
                    contLoop = true;
                    break;
                }
            }
        }
    }
    
    /**
     * @brief Update training set for a specific label
     * @param label label of the sub-training set to update
     * @throws out_of_range if the label does not exist
     */
    virtual void updateTrainingSet(Label const& label)
    {
        if (globalTrainingSet->allLabels.find(label) == globalTrainingSet->allLabels.end())
            throw out_of_range("Class " + label.as_string() + " does not exist");
        
        if (models.find(label) == models.end()) {
            models[label] = referenceModel_;
        }
        
        TrainingSet* new_ts = globalTrainingSet->getSubTrainingSetForClass(label);
        models[label].set_trainingSet(new_ts);
        models[label].trained = false;
    }
    
    /**
     * @brief Update the training set for all labels
     * @details Checks for deleted classes, and tracks modifications of the training sets associated with each class
     */
    virtual void updateAllTrainingSets()
    {
        removeDeprecatedModels();
        if (!globalTrainingSet->has_changed()) return;
        
        // Update classes models and training sets
        for (typename set<Label>::iterator it=globalTrainingSet->allLabels.begin(); it != globalTrainingSet->allLabels.end(); ++it)
        {
            updateTrainingSet(*it);
        }
        
        globalTrainingSet->set_unchanged();
    }
    
#pragma mark -
#pragma mark === Protected attributes ===
    /**
     * @brief defines if the phrase is bimodal (true) or unimodal (false)
     */
    bool bimodal_;
    
    /**
     * @brief Playing mode
     * @see POLYPLAYMODE
     */
    GROUP_ESTIMATION_MODE performanceMode_;
    
    /**
     * @brief reference model, Used to store shared model attributes
     */
    ModelType referenceModel_;
    
    /**
     * @brief Callback function for the training algorithm
     */
    void (*trainingCallbackFunction_)(void *srcModel, CALLBACK_FLAG state, void* extradata);
    
    /**
     * @brief Extra data to pass in argument to the callback function
     */
    void *trainingExtradata_;
    
    /**
     * @brief Number of Models that are still training
     */
    unsigned int models_to_train_;
    
#ifdef USE_PTHREAD
    /**
     * @brief Training Threads
     * @warning only defined if USE_PTHREAD is defined
     */
    map<Label, pthread_t> training_threads;
#endif
};

#endif
