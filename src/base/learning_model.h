//
// learning_model.h
//
// Base class for machine learning models. Real-time implementation and interface.
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef rtml_learning_model_h
#define rtml_learning_model_h

#include "training_set.h"
#include "listener.h"

template <typename phraseType> class TrainingSet;

typedef enum CALLBACK_FLAG__
{
    TRAINING_RUN,
    TRAINING_DONE,
    TRAINING_ERROR
} CALLBACK_FLAG;

#pragma mark -
#pragma mark Class Definition
/*!
 @class LearningModel
 @brief Base class for Machine Learning models
 @todo class description
 @tparam phraseType Data type of the phrases composing the training set
 */
template <typename phraseType>
class LearningModel : public Listener {
public:
    bool trained;
    TrainingSet<phraseType> *trainingSet;
    
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors */
    /*!
     Constructor
     @param _trainingSet training set on wich the model is trained
     */
    LearningModel(TrainingSet<phraseType> *_trainingSet)
    {
        trained = false;
        trainingSet = _trainingSet;
        trainingCallback = NULL;
    }
    
    /*!
     Copy constructor
     */
    LearningModel(LearningModel<phraseType> const& src)
    {
        this->_copy(this, src);
    }
    
    /*!
     Assignment
     */
    LearningModel<phraseType>& operator=(LearningModel<phraseType> const& src)
    {
        if(this != &src)
        {
            _copy(this, src);
        }
        return *this;
    }
    
    /*!
     Copy between to models (called by copy constructor and assignment methods)
     */
    virtual void _copy(LearningModel<phraseType> *dst,
                       LearningModel<phraseType> const& src)
    
    {
        dst->trained = src.trained;
        dst->trainingSet = src.trainingSet;
        dst->trainingCallback = src.trainingCallback;
    }
    
    /*!
     destructor
     */
    virtual ~LearningModel()
    {}
    
#pragma mark -
#pragma mark Connect Training set
    /*! @name training set */
    /*!
     set the training set associated with the model
     */
    void set_trainingSet(TrainingSet<phraseType> *_trainingSet)
    {
        trainingSet = _trainingSet;
    }

#pragma mark -
#pragma mark Callback function for training
    /*! @name training set */
    /*!
     set the training set associated with the model
     */
    void set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata) {
        this->trainingExtradata = extradata;
        this->trainingCallback = callback;
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    /*!
     write model to stream
     @param outStream output stream
     */
    virtual void write(ostream& outStream)
    {
    }
    
    /*!
     read model from stream
     @param inStream input stream
     */
    virtual void read(istream& inStream)
    {
    }
    
#pragma mark -
#pragma mark Pure Virtual Methods: Training, Playing
    /*! @name Pure virtual methods */
    virtual void initTraining() = 0;
    virtual int train() = 0;
    virtual void finishTraining()
    {
        if (this->trainingCallback)
            this->trainingCallback(this, TRAINING_DONE, this->trainingExtradata);
    }
    virtual void initPlaying()
    {
        if (!this->trained) {
            throw RTMLException("Cannot play: model has not been trained", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    virtual double play(float *obs) = 0;
    
    float trainingProgression;
protected:
    void (*trainingCallback)(void *srcModel, CALLBACK_FLAG state, void* extradata);
    void *trainingExtradata;
};

#endif