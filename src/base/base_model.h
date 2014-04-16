//
// learning_model.h
//
// Base class for machine learning models. Real-time implementation and interface.
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
//

#ifndef rtml_learning_model_h
#define rtml_learning_model_h

#include "training_set.h"

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
 * @defgroup ModelBase Base Classes for Probabilistic Models
 */

/**
 * @ingroup ModelBase
 * @class BaseModel
 * @brief Base class for Machine Learning models
 * @details both unimodal and multimodal (specified in Constructor flags)
 * @todo class description
 */
class BaseModel : public Listener {
public:
    friend class GMMGroup;
    friend class HierarchicalHMM;
        
#pragma mark -
#pragma mark === Public Interface ===
#pragma mark > Constructors
    /*@{*/
    /** @name Constructors */
    /**
     * @brief Constructor
     * @param trainingSet training set on which the model is trained
     * @param flags Construction Flags. BIMODAL indicates that the model should be used for regression
     * (bimodal model). To use in conjunction with a bimodal training set.
     */
    BaseModel(rtml_flags flags = NONE,
                  TrainingSet *trainingSet = NULL);
    
    /**
     * @brief Copy constructor
     * @param src Source model
     */
    BaseModel(BaseModel const& src);
    
    /**
     * @brief Assignment
     * @param src Source model
     */
    BaseModel& operator=(BaseModel const& src);
    
    /**
     * @brief destructor
     */
    virtual ~BaseModel();
    
    /*@}*/

#pragma mark > Training set
    /*@{*/
    /** @name Training set */
    /**
     * @brief set the training set associated with the model
     * @details updates the dimensions of the model
     * @param trainingSet pointer to the training set.
     * @throws runtime_error if the training set has not the same number of modalities
     */
    void set_trainingSet(TrainingSet *trainingSet);
    
    /**
     * @brief handle notifications of the training set
     * @details here only the dimensions attributes of the training set are considered
     * @param attribute name of the attribute: should be either "dimension" or "dimension_input"
     */
    void notify(string attribute);
    
    /*@}*/

#pragma mark > Accessors
    /*@{*/
    /** @name Accessors */
    /**
     * @brief Get Total Dimension of the model (sum of dimension of modalities)
     * @return total dimension of Gaussian Distributions
     */
    int get_dimension() const
    {
        return dimension_;
    }
    
    /**
     * @brief Get the dimension of the input modality
     * @warning This can only be used in bimodal mode (construction with 'BIMODAL' flag)
     * @return dimension of the input modality
     * @throws runtime_error if not in bimodal mode
     */
    int get_dimension_input() const
    {
        if (!bimodal_)
            throw runtime_error("Phrase is not Bimodal");
        return dimension_input_;
    }

    /*@}*/

#pragma mark > Callback function for training
    /*@{*/
    /** @name Callback function for training */
    /**
     * @brief set the callback function associated with the training algorithm
     * @details the function is called whenever the training is over or an error happened during training
     */
    void set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata);
    
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

#pragma mark > Pure Virtual Methods: Allocation, Training, Playing
    /*@{*/
    /** @name Pure Virtual Methods: Allocation, Training, Playing */
    /**
     * @brief Allocate memory for the model's parameters
     * @details called when dimensions are modified
     */
    virtual void allocate() = 0;
    
    /**
     * @brief Initialize the training algorithm
     * @todo Put this in EMBasedModel and rename to train_EM_init()
     */
    virtual void initTraining() = 0;

    /**
     * @brief Main Training Function
     * @return number of iteration of the training process
     */
    virtual int train() = 0;

    /**
     * @brief Terminate the training algorithm
     */
    virtual void finishTraining();

    /**
     * @brief Initialize the 'Performance' phase: prepare model for playing.
     */
    virtual void initPlaying();

    /**
     * @brief Main Play function: updates the predictions of the model given a new observation
     * @param observation observation vector (must be of size 'dimension' or 'dimension_input' 
     * depending on the mode [unimodal/bimodal])
     * @return likelihood of the observation
     */
    virtual double play(vector<float> const& observation) = 0;
    
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
    
protected:
#pragma mark -
#pragma mark === Protected Methods ===
#pragma mark > Copy between models
    /*@{*/
    /** @name Copy between models */
    /**
     * @brief Copy between to models (called by copy constructor and assignment methods)
     * @param src Source model
     * @param dst Destination model
     */
    virtual void _copy(BaseModel *dst, BaseModel const& src);

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
    int dimension_;
    
    /**
     * @brief Dimension of the input modality
     */
    int dimension_input_;
    
    /**
     * @brief Callback function for the training algorithm
     */
    void (*trainingCallback_)(void *srcModel, CALLBACK_FLAG state, void* extradata);

    /**
     * @brief Extra data to pass in argument to the callback function
     */
    void *trainingExtradata_;
};

#endif