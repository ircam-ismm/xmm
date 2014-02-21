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
class ConcurrentModels : public Listener
{
public:
    /*!
     @enum POLYPLAYMODE
     type of playing mode for concurrent models
     */
    enum POLYPLAYMODE {
        LIKELIEST = 0, //<! the play method returns the results of the likeliest model
        MIXTURE = 1    //<! the play method returns a weighted sum of the results of each model
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
#pragma mark Model Utilities
    /*!
     Remove All models
     */
    virtual void clear()
    {
        models.clear();
    }
    
    /*!
     Remove Specific model
     @param classLabel class label of the model
     @throw RTMLException if the class does not exist
     */
    virtual void remove(Label classLabel)
    {
        model_iterator it = models.find(classLabel);
        if (it == models.end())
            throw RTMLException("Class Label Does not exist", __FILE__, __FUNCTION__, __LINE__);
        it->second.initTraining();
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
            throw RTMLException("Class " + classLabel.as_string() + " Does not exist", __FILE__, __FUNCTION__, __LINE__);
        models[classLabel].initTraining();
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
        updateTrainingSet(classLabel);
        this->initTraining(classLabel);
#if __cplusplus > 199711L
        thread (&ModelType::train, &models[classLabel]).detach();
        return 0;
#else
        return models[classLabel].train();
#endif
    }
    
    /*!
     Train All model which data has changed.
     */
    virtual map<Label, int> retrain()
    {
        updateAllTrainingSets();
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
        updateAllTrainingSets();
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
     Remove models which label is not in the training set anymore
     */
    virtual void removeDeprecatedModels()
    {
        if (globalTrainingSet->is_empty()) return;
        
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
    }
    
    /*!
     Update training set for a specific model
     @param label label of the sub-training set to update
     */
    virtual void updateTrainingSet(Label label)
    {
        if (globalTrainingSet->is_empty()) return;
        if (globalTrainingSet->allLabels.find(label) == globalTrainingSet->allLabels.end())
            throw RTMLException("Class " + label.as_string() + " Does not exist", __FILE__, __FUNCTION__, __LINE__);
            
        if (models.find(label) == models.end()) {
            models[label] = referenceModel;
            models[label].trainingSet = NULL;
        }
        
        TrainingSet<phraseType> *model_ts = models[label].trainingSet;
        TrainingSet<phraseType> *new_ts = (TrainingSet<phraseType> *)globalTrainingSet->getSubTrainingSetForClass(label);
        if (!model_ts || *model_ts != *new_ts) {
            if (model_ts)
                delete model_ts;
            models[label].set_trainingSet(new_ts);
            new_ts->set_parent(&models[label]);
            
            models[label].trained = false;
        }
    }
    
    /*!
     Update the training set for each model
     
     Checks for deleted classes, and tracks modifications of the training sets associated with each class
     */
    virtual void updateAllTrainingSets()
    {
        if (globalTrainingSet->is_empty()) return;
        if (!globalTrainingSet->has_changed()) return;
        
        removeDeprecatedModels();
        
        // Update classes models and training sets
        for (typename set<Label>::iterator it=globalTrainingSet->allLabels.begin(); it != globalTrainingSet->allLabels.end(); it++)
        {
            updateTrainingSet(*it);
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
#pragma mark Playing
#pragma mark -
#pragma mark Playing
    /*! @name Playing */
    /*!
     Initialize Playing
     */
    virtual void initPlaying()
    {
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.initPlaying();
        }
    }
    
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
    /*!
     Write to JSON Node
     */
    virtual JSONNode to_json() const
    {
        JSONNode json_ccmodels(JSON_NODE);
        json_ccmodels.set_name("Concurrent Models");
        json_ccmodels.push_back(JSONNode("size", models.size()));
        json_ccmodels.push_back(JSONNode("playmode", int(playMode)));
        
        // Add reference model
        JSONNode json_refModel = referenceModel.to_json();
        json_refModel.set_name("reference model");
        json_ccmodels.push_back(json_refModel);
        
        // Add phrases
        JSONNode json_models(JSON_ARRAY);
        for (const_model_iterator it = models.begin(); it != models.end(); it++)
        {
            JSONNode json_model(JSON_NODE);
            json_model.push_back(it->first.to_json());
            json_model.push_back(it->second.to_json());
            json_models.push_back(json_model);
        }
        json_models.set_name("models");
        json_ccmodels.push_back(json_models);
        
        return json_ccmodels;
    }
    
    /*!
     Read from JSON Node
     */
    virtual void from_json(JSONNode root)
    {
        try {
            assert(root.type() == JSON_NODE);
            JSONNode::const_iterator root_it = root.begin();
            
            // Get Size: Number of Models
            assert(root_it != root.end());
            assert(root_it->name() == "size");
            assert(root_it->type() == JSON_NUMBER);
            int numModels = root_it->as_int();
            root_it++;
            
            // Get Play Mode
            assert(root_it != root.end());
            assert(root_it->name() == "playmode");
            assert(root_it->type() == JSON_NUMBER);
            playMode = (root_it->as_int() > 0) ? MIXTURE : LIKELIEST;
            root_it++;
            
            // Get Reference Model
            assert(root_it != root.end());
            assert(root_it->name() == "reference model");
            assert(root_it->type() == JSON_NODE);
            referenceModel.from_json(*root_it);
            root_it++;
            
            // Get Phrases
            models.clear();
            assert(root_it != root.end());
            assert(root_it->name() == "models");
            assert(root_it->type() == JSON_ARRAY);
            for (int i=0 ; i<numModels ; i++)
            {
                // Get Label
                JSONNode::const_iterator array_it = (*root_it)[i].begin();
                assert(array_it != root_it->end());
                assert(array_it->name() == "label");
                assert(array_it->type() == JSON_NODE);
                Label l;
                l.from_json(*array_it);
                array_it++;
                
                // Get Phrase Content
                assert(array_it != root_it->end());
                assert(array_it->type() == JSON_NODE);
                models[l] = this->referenceModel;
                models[l].trainingSet = NULL;
                models[l].from_json(*array_it);
            }
            
            assert(numModels == models.size());
            
        } catch (exception &e) {
            throw RTMLException("Error reading JSON, Node: " + root.name() + " >> " + e.what());
        }
    }
    
    
    virtual void write(ostream& outStream)
    {
        outStream << "# CONCURRENT MODELS\n";
        outStream << "# ======================================\n";
        outStream << "# Number of models\n";
        outStream << models.size() << endl;
        outStream << "# Play Mode\n";
        outStream << playMode << endl;
        outStream << "# Reference Model\n";
        referenceModel.initParametersToDefault();
        referenceModel.write(outStream);
        outStream << "# === MODELS\n";
        for (model_iterator it = models.begin(); it != models.end(); it++) {
            outStream << "# Model Label\n";
            if (it->first.type == Label::INT)
                outStream << "INT " << it->first.getInt() << endl;
            else
                outStream << "SYM " << it->first.getSym() << endl;
            it->second.write(outStream);
        }
    }
    
    virtual void read(istream& inStream)
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
        this->referenceModel.read(inStream);
        
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
            this->models[lab].read(inStream);
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
