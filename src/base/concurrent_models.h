//
// concurrent_models.h
//
// Multiple machine learning models running in parallel
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef rtml_concurrent_models_h
#define rtml_concurrent_models_h

#include "training_set.h"
#include "learning_model.h"
#if __cplusplus > 199711L
#include <thread>
#endif

using namespace std;

#pragma mark -
#pragma mark Class Definition
/*!
 @class ConcurrentModels
 @brief Handle concurrent machine learning models running in parallel
 @todo class description
 @tparam modelType type of the models
 @tparam phraseType type of the phrase in the training set (@see Phrase, MultimodalPhrase, GestureSoundPhrase)
 */
template<typename ModelType, typename phraseType>
class ConcurrentModels : public Notifiable
{
public:
    /*!
     @enum POLYPLAYMODE
     type of playing mode for concurrent models
     */
    enum POLYPLAYMODE {
        LIKELIEST, //<! the play method returns the results of the likeliest model
        MIXTURE    //<! the play method returns a weighted sum of the results of each model
    };
    
    typedef typename  map<Label, ModelType>::iterator model_iterator;
    typedef typename  map<Label, ModelType>::const_iterator const_model_iterator;
    typedef typename  map<int, Label>::iterator labels_iterator;
    
    map<Label, ModelType> models;
    TrainingSet<phraseType> *globalTrainingSet;    //<! Global training set: contains all phrases (all labels)
    
#pragma mark -
#pragma mark Constructors
    /*! name Constructors */
    /*!
     Constructor
     @param _globalTrainingSet global training set: contains all phrases for each model
     */
    ConcurrentModels(TrainingSet<phraseType> *_globalTrainingSet=NULL)
    {
        globalTrainingSet = _globalTrainingSet;
        if (globalTrainingSet)
            globalTrainingSet->set_parent(this);
        referenceModel.set_trainingSet(globalTrainingSet);
        playMode = LIKELIEST;
    }
    
    /*!
     Destructor
     */
    virtual ~ConcurrentModels()
    {
        for (model_iterator it=models.begin(); it != models.end(); it++) {
            delete it->second.trainingSet;
        }
        models.clear();
    }
    
#pragma mark -
#pragma mark Notifications
    /*! @name Notifications */
    /*!
     Receives notifications from the global training set and dispatches to sub training sets
     */
    virtual void notify(string attribute)
    {
        referenceModel.notify(attribute);
        for (model_iterator it=models.begin(); it != models.end(); it++) {
            it->second.notify(attribute);
        }
    }
    
    
#pragma mark -
#pragma mark Accessors
    /*! @name Accessors */
    /*!
     Set pointer to the global training set
     @param _globalTrainingSet pointer to the global training set
     */
    void set_trainingSet(TrainingSet<phraseType> *_globalTrainingSet)
    {
        globalTrainingSet = _globalTrainingSet;
        if (globalTrainingSet)
            globalTrainingSet->set_parent(this);
        referenceModel.set_trainingSet(globalTrainingSet);
    }
    
    /*!
     Check if a model has been trained
     @param classLabel class label of the model
     @throw RTMLException if the class does not exist
     */
    bool is_trained(Label classLabel)
    {
        if (models.find(classLabel) == models.end())
            throw RTMLException("Class Label Does not exist", __FILE__, __FUNCTION__, __LINE__);
        return models[classLabel].trained;
    }
    
    /*
     bool is_trained(int intLabel)
     {
     Label l;
     l.setInt(intLabel);
     return is_trained(l);
     }
     
     bool is_trained(string symLabel)
     {
     Label l;
     l.setSym(symLabel);
     return is_trained(l);
     }
     //*/
    
    /*!
     Check if all models have been trained
     */
    bool is_trained()
    {
        if (this->size() == 0)
            return false;
        for (model_iterator it = models.begin(); it != models.end(); it++) {
            if (!it->second.trained)
                return false;
        }
        return true;
    }
    
    /*!
     get size: number of models
     */
    unsigned int size() const
    {
        return (unsigned int)(models.size());
    }
    
#pragma mark -
#pragma mark Training
    /*! @name Training */
    /*!
     Initialize training for given model
     @param classLabel class label of the model
     @throw RTMLException if the class does not exist
     */
    virtual void initTraining(Label classLabel)
    {
        model_iterator it = models.find(classLabel);
        if (it == models.end())
            throw RTMLException("Class Label Does not exist", __FILE__, __FUNCTION__, __LINE__);
        it->second.initTraining();
    }
    
    /*!
     Initialize training for each model
     */
    virtual void initTraining()
    {
        for (model_iterator it = models.begin(); it != models.end(); it++) {
            it->second.initTraining();
        }
    }
    
    /*!
     Train 1 model. The model is trained even if the dataset has not changed
     @param classLabel class label of the model
     @throw RTMLException if the class does not exist
     */
    virtual int train(Label classLabel)
    {
        updateTrainingSets();
        if (models.find(classLabel) == models.end())
            throw RTMLException("Class Label Does not exist", __FILE__, __FUNCTION__, __LINE__);
        
        return models[classLabel].train();
    }
    
    /*!
     Train All model which data has changed.
     */
    virtual map<Label, int> retrain()
    {
        updateTrainingSets();
        map<Label, int> nbIterations;
        
        RTMLException trainingException;
        bool trainingFailed(false);
        
        for (model_iterator it=models.begin(); it != models.end(); it++) {
            nbIterations[it->first] = 0;
            if (!it->second.trained || it->second.trainingSet->has_changed()) {
                it->second.initTraining();
                try {
                    nbIterations[it->first] = it->second.train();
                } catch (RTMLException &e) {
                    trainingException = e;
                    trainingFailed = true;
                }
            }
        }
        
        if (trainingFailed) {
            throw trainingException;
        }
        
        return nbIterations;
    }
    
    /*!
     Train All model even if their data has not changed.
     */
    virtual map<Label, int> train()
    {
        updateTrainingSets();
        this->initTraining();
        map<Label, int> nbIterations;
        
#if __cplusplus > 199711L
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            thread (&ModelType::train, &it->second).detach();
        }
#else
        // Sequential training
        for (model_iterator it=models.begin(); it != models.end(); it++) {
            nbIterations[it->first] = it->second.train();
        }
#endif
        return nbIterations;
    }
    
    void set_trainingCallback(void (*callback)(void *srcModel, CALLBACK_FLAG state, void* extradata), void* extradata) {
        this->referenceModel.set_trainingCallback(callback, extradata);
        for (model_iterator it=models.begin(); it != models.end(); it++) {
            it->second.set_trainingCallback(callback, extradata);
        }
    }
    
    /*!
     Finish training for given model
     @param classLabel class label of the model
     @throw RTMLException if the class does not exist
     */
    virtual void finishTraining(Label classLabel)
    {
        model_iterator it = models.find(classLabel);
        if (it == models.end())
            throw RTMLException("Class Label Does not exist", __FILE__, __FUNCTION__, __LINE__);
        it->second.finishTraining();
    }
    
    /*!
     Finish training for each model
     */
    virtual void finishTraining()
    {
        for (model_iterator it = models.begin(); it != models.end(); it++) {
            it->second.finishTraining();
        }
    }
    
#pragma mark -
#pragma mark Training Set
    /*! @name Handle Training set */
    /*!
     Update training set for each model
     
     Checks for deleted classes, and tracks modifications of the training sets associated with each class
     */
    virtual void updateTrainingSets()
    {
        if (globalTrainingSet->is_empty()) return;
        if (!globalTrainingSet->has_changed()) return;
        
        // Look for deleted classes
        bool contLoop(true);
        while (contLoop) {
            contLoop = false;
            for (model_iterator it = models.begin(); it != models.end(); it++) {
                if (globalTrainingSet->allLabels.find(it->first) == globalTrainingSet->allLabels.end())
                {
                    delete models[it->first].trainingSet;
                    models.erase(it->first);
                    contLoop = true;
                    break;
                }
            }
        }
        
        // Update classes models and training sets
        for (typename set<Label>::iterator it=globalTrainingSet->allLabels.begin(); it != globalTrainingSet->allLabels.end(); it++)
        {
            // TODO: problem ==> no indication of changes in data
            if (models.find(*it) == models.end()) {
                models[*it] = referenceModel;
                models[*it].trainingSet = NULL;
            }
            
            TrainingSet<phraseType> *model_ts = models[*it].trainingSet;
            TrainingSet<phraseType> *new_ts = (TrainingSet<phraseType> *)globalTrainingSet->getSubTrainingSetForClass(*it);
            if (!model_ts || *model_ts != *new_ts) {
                if (model_ts)
                    delete model_ts;
                models[*it].set_trainingSet(new_ts);
                new_ts->set_parent(&models[*it]);
                
                models[*it].trained = false;
            }
        }
        
        globalTrainingSet->set_unchanged();
    }
    
#pragma mark -
#pragma mark Play Mode
    /*! @name Play Mode */
    void set_playMode(string playMode_str)
    {
        if (!playMode_str.compare("likeliest")) {
            playMode = LIKELIEST;
        } else if (!playMode_str.compare("mixture")) {
            playMode = MIXTURE;
        } else {
            throw RTMLException("Unknown playing mode for multiple models", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    
    string get_playMode()
    {
        if (playMode == LIKELIEST)
            return "likeliest";
        else
            return "mixture";
    }
    
#pragma mark -
#pragma mark Play! ==> Pure virtual method
    /*! @name Play /// Pure virtual */
    /*!
     Play method for multiple models
     @param obs multimodal observation vector
     @param modelLikelihoods array to contain the likelihood of each
     */
    virtual void play(float *obs, double *modelLikelihoods) = 0;
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    virtual void write(ostream& outStream, bool writeTrainingSet=false)
    {
        outStream << "# CONCURRENT MODELS\n";
        outStream << "# ======================================\n";
        outStream << "# Number of models\n";
        outStream << models.size() << endl;
        outStream << "# Play Mode\n";
        outStream << playMode << endl;
        outStream << "# Reference Model\n";
        referenceModel.initParametersToDefault();
        referenceModel.write(outStream, false);
        outStream << "# === MODELS\n";
        for (model_iterator it = models.begin(); it != models.end(); it++) {
            outStream << "# Model Label\n";
            if (it->first.type == Label::INT)
                outStream << "INT " << it->first.getInt() << endl;
            else
                outStream << "SYM " << it->first.getSym() << endl;
            it->second.write(outStream);
        }
        if (writeTrainingSet)
            globalTrainingSet->write(outStream);
    }
    
    virtual void read(istream& inStream, bool readTrainingSet=false)
    {
        // Get number of models
        skipComments(&inStream);
        int nbModels;
        inStream >> nbModels;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        for (model_iterator it=models.begin(); it != models.end(); it++) {
            delete it->second.trainingSet;
        }
        models.clear();
        
        // Get playing mode
        skipComments(&inStream);
        int _playmmode;
        inStream >> _playmmode;
        playMode = POLYPLAYMODE(_playmmode);
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        // Read Reference Model
        this->referenceModel.read(inStream, false);
        
        // Read Models
        // TODO: I guess the Label Does NOT work
        for (int i=0; i<nbModels; i++) {
            // Read label
            skipComments(&inStream);
            Label lab;
            string lType;
            inStream >> lType;
            if (!inStream.good())
                throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
            if (lType == "INT") {
                int intLab;
                inStream >> intLab;
                if (!inStream.good())
                    throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
                lab.setInt(intLab);
            } else {
                string symLab;
                inStream >> symLab;
                if (!inStream.good())
                    throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
                lab.setSym(symLab);
            }
            this->models[lab].read(inStream, false);
        }
        
        // Read Training set
        if (readTrainingSet) {
            globalTrainingSet->read(inStream);
            updateTrainingSets();
        }
    }
    
#ifdef SWIGPYTHON
    void printLabels() {
        cout << "Order of Labels: ";
        for (model_iterator it = this->models.begin() ; it != this->models.end() ; it++)
            if (it->first.type == Label::INT) {
                cout << it->first.getInt() << " ";
            }
            else {
                cout << it->first.getSym() << " ";
            }
        cout << endl;
    }
#endif
    
#pragma mark -
#pragma mark Protected attributes
    /*! @name Protected attributes */
protected:
    POLYPLAYMODE playMode;
    ModelType referenceModel; // Used to store shared model attributes
};

#endif
