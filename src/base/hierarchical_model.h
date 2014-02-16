//
// hierarchical_model.h
//
// Hierarchical Model => used for Hierarchical HMM implementation
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef mhmm_hierarchical_model_h
#define mhmm_hierarchical_model_h

#include "concurrent_models.h"

using namespace std;


const double HIERARCHICALMODEL_DEFAULT_EXITTRANSITION = 0.1;
const bool HIERARCHICALMODEL_DEFAULT_INCREMENTALLEARNING = false;

#pragma mark -
#pragma mark Class Definition
/*!
 @class HierarchicalModel
 @brief Handle Hierarchical machine learning models (specific to HMMs)
 @todo class description
 @tparam modelType type of the models
 @tparam phraseType type of the phrase in the training set (@see Phrase, MultimodalPhrase, GestureSoundPhrase)
 @tparam Label type of the labels for each class.
 */
template<typename ModelType, typename phraseType>
class HierarchicalModel
: public ConcurrentModels<ModelType, phraseType> {
public:
    /*! @name iterators */
    typedef typename  map<Label, ModelType>::iterator model_iterator;
    typedef typename  map<Label, ModelType>::const_iterator const_model_iterator;
    typedef typename  map<int, Label>::iterator labels_iterator;
    
    map<Label, double> prior;
    map<Label, double> exitTransition;
    map<Label, map<Label, double> > transition;
    
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors */
    HierarchicalModel(TrainingSet<phraseType> *_globalTrainingSet=NULL)
    : ConcurrentModels<ModelType, phraseType>(_globalTrainingSet)
    {
        incrementalLearning = HIERARCHICALMODEL_DEFAULT_INCREMENTALLEARNING;
    }
    
    virtual ~HierarchicalModel()
    {
        prior.clear();
        transition.clear();
        exitTransition.clear();
    }
    
#pragma mark -
#pragma mark Accessors
    /*! @name Accessors */
    /*!
     * @brief return learning mode: "incremental" or "ergodic"
     *
     * @return learningMode "incremental" if incrementalLearning == true
     */
    string get_learningMode() const
    {
        string learningMode;
        if (incrementalLearning) {
            learningMode = "incremental";
        } else {
            learningMode = "ergodic";
        }
        return learningMode;
    }
    
    /*!
     * @brief set learning mode: "incremental" or "ergodic"
     *
     * @param learningMode "incremental" sets 'incrementalLearning == true' / "ergodic"
     */
    void set_learningMode(string learningMode)
    {
        if (learningMode == "incremental") {
            incrementalLearning = true;
        } else { // if (learningMode == "ergodic")
            incrementalLearning = false;
        }
    }
    
    /*!
     * @brief get high-level Prior probabilities
     * @return High-level prior probability vector (double array)
     */
    double* get_prior() const
    {
        double *prior_ = new double[this->size()];
        int l(0);
        for (const_model_iterator it = this->models.begin(); it != this->models.end(); it++) {
            prior_[l++] = this->prior.at(it->first);
        }
        return prior_;
    }
    /*!
     * @brief set prior probabilities
     * @param prior_ high-level probability vector (size nbPrimitives)
     */
    void set_prior(double *prior_){
        try {
            int l(0);
            for (model_iterator it = this->models.begin() ; it != this->models.end() ; it++) {
                this->prior[it->first] = max(prior_[l++], 0.0);
            }
            this->normalizeTransitions();
        } catch (exception &e) {
            throw RTMLException("Wrong format for prior", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    
    // High-level Transition Matrix
    /*!
     * @brief get high-level transition matrix
     * @return High-level transition matrix (2D array double)
     */
    double* get_transition() const
    {
        unsigned int nbPrimitives = this->size();
        double *trans_ = new double[nbPrimitives*nbPrimitives];
        int l(0);
        
        for (const_model_iterator srcit = this->models.begin(); srcit != this->models.end(); srcit++) {
            for (const_model_iterator dstit = this->models.begin(); dstit != this->models.end(); dstit++) {
                trans_[l++] = this->transition.at(srcit->first).at(dstit->first);
            }
        }
        return trans_;
    }
    /*!
     * @brief set transition matrix
     * @param trans_ high-level transition matrix (2D array double)
     */
    void set_transition(double *trans_) {
        try {
            int l(0);
            for (model_iterator srcit = this->models.begin(); srcit != this->models.end(); srcit++) {
                for (model_iterator dstit = this->models.begin(); dstit != this->models.end(); dstit++) {
                    this->transition[srcit->first][dstit->first] = max(trans_[l++], 0.0);
                }
            }
            this->normalizeTransitions();
        } catch (exception &e) {
            throw RTMLException("Wrong format for transition", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    
    /*!
     @return the exit transition vector of the high level
     */
    double* get_exitTransition() const
    {
        double *exittrans_ = new double[this->size()];
        int l(0);
        
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; it++)
        {
            exittrans_[l++] = this->exitTransition.at(it->first);
        }
        return exittrans_;
    }
    
    /*!
     set the exit transition vector of the high level
     */
    void set_exitTransition(double *exittrans_)
    {
        try {
            int l(0);
            for (model_iterator it = this->models.begin() ; it != this->models.end() ; it++) {
                this->exitTransition[it->first] = max(exittrans_[l++], 0.0);
            }
            this->normalizeTransitions();
        } catch (exception &e) {
            throw RTMLException("Wrong format for prior", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    
#pragma mark -
#pragma mark Transition parameters
    /*! @name High level transition parameters */
    /*!
     * Normalize segment level prior and transition matrices
     */
    void normalizeTransitions()
    {
        double sumPrior(0.0);
        for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; srcit++)
        {
            sumPrior += prior[srcit->first];
            double sumTrans(0.0);
            for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; dstit++)
                sumTrans += transition[srcit->first][dstit->first];
            for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; dstit++)
                transition[srcit->first][dstit->first] /= sumTrans;
        }
        for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; srcit++)
            prior[srcit->first] /= sumPrior;
    }
    
    /*!
     * @brief set a particular value of the transition matrix
     *
     * set trans(i,j) = proba
     * @param srcSegmentLabel origin segment
     * @param dstSegmentLabel target segment
     * @param proba probability of making a transition from srcSegmentLabel to dstSegmentLabel
     */
    void setOneTransition(Label srcSegmentLabel, Label dstSegmentLabel, double proba)
    {
        transition[srcSegmentLabel][dstSegmentLabel] = min(proba, 1.);
        normalizeTransitions();
        // TODO: absolute/relative mode?
    }
    
#pragma mark -
#pragma mark High level parameters: update and estimation
    /*! @name High level parameters: update and estimation */
    /*!
     * @brief update high-level parameters when a new primitive is learned
     *
     * updated parameters: prior probabilities + transition matrix
     */
    void updateTransitionParameters()
    {
        if (this->size() == prior.size()) // number of primitives has not changed
            return;
        
        if (incrementalLearning) {          // incremental learning: use regularization to preserve transition
            updatePrior_incremental();
            updateTransition_incremental();
        } else {                            // ergodic learning: set ergodic prior and transition probabilities
            updatePrior_ergodic();
            updateTransition_ergodic();
        }
        
        updateExitProbabilities(); // Update exit probabilities of Submodels (signal level)
    }
    
    /*!
     * @brief incremental learning: update high-level prior probabilities (regularization)
     */
    void updatePrior_incremental()
    {
        int oldNbPrim = prior.size();
        int regularizationFactor = 1;
        int nbPrimitives = this->size();
        
        if (oldNbPrim>0)
        {
            for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; it++)
                if (prior.find(it->first) == prior.end())
                {
                    prior[it->first] += double(regularizationFactor);
                    prior[it->first] /= double(nbPrimitives + regularizationFactor) ;
                } else {
                    prior[it->first] = 1. / double(nbPrimitives + regularizationFactor);
                }
        } else {
            for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; it++)
                prior[it->first] = 1. / double(nbPrimitives);
        }
        
    }
    
    /*!
     * @brief incremental learning: update high-level transition matrix  (regularization)
     *
     * (transition probabilities between primitive gestures)
     */
    void updateTransition_incremental()
    {
        int oldNbPrim = prior.size();
        int regularizationFactor = 1;
        int nbPrimitives = this->size();
        
        if (oldNbPrim>0)
        {
            map<Label, map<Label, double> > oldTransition = transition;;
            
            for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; srcit++)
            {
                for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; dstit++)
                {
                    if (transition.find(srcit->first) == transition.end() || transition[srcit->first].find(dstit->first) == transition[srcit->first].end())
                    {
                        transition[srcit->first][dstit->first] = 1/double(nbPrimitives+regularizationFactor);
                    } else {
                        transition[srcit->first][dstit->first] += double(regularizationFactor);
                        transition[srcit->first][dstit->first] /= double(nbPrimitives+regularizationFactor);
                    }
                }
                
                if (exitTransition.find(srcit->first) == exitTransition.end())
                {
                    exitTransition[srcit->first] = 1/double(nbPrimitives+regularizationFactor);
                } else {
                    exitTransition[srcit->first] += double(regularizationFactor);
                    exitTransition[srcit->first] /= double(nbPrimitives+regularizationFactor);
                }
            }
        } else {
            for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; srcit++)
            {
                exitTransition[srcit->first] = HIERARCHICALMODEL_DEFAULT_EXITTRANSITION;
                
                for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; dstit++)
                    transition[srcit->first][dstit->first] = 1/(double)nbPrimitives;
            }
        }
    }
    
    /*!
     * @brief ergodic learning update high-level prior probabilities -> equal prior probs
     */
    void updatePrior_ergodic()
    {
        int nbPrimitives = this->size();
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; it++)
            prior[it->first] = 1/double(nbPrimitives);
    }
    
    /*!
     * @brief ergodic learning: update high-level transition matrix
     *
     * (equal transition probabilities between primitive gestures)
     */
    void updateTransition_ergodic()
    {
        int nbPrimitives = this->size();
        for (const_model_iterator srcit = this->models.begin() ; srcit != this->models.end() ; srcit++)
        {
            exitTransition[srcit->first] = HIERARCHICALMODEL_DEFAULT_EXITTRANSITION;
            for (const_model_iterator dstit = this->models.begin() ; dstit != this->models.end() ; dstit++)
                transition[srcit->first][dstit->first] =  1/double(nbPrimitives);
        }
    }
    
    void updateExitProbabilities()
    {
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.updateExitProbabilities();
        }
    }
    
    virtual void updateTrainingSets()
    {
        ConcurrentModels<ModelType, phraseType>::updateAllTrainingSets();
        updateTransitionParameters();
    }
    
#pragma mark -
#pragma mark Training
    /*! @name Training */
    /*!
     Initialize training for each model
     */
    virtual void initTraining()
    {
        ConcurrentModels<ModelType, phraseType>::initTraining();
        updateTransitionParameters();
    }
    
    /*!
     Train 1 model. The model is trained even if the dataset has not changed
     @param classLabel class label of the model
     @throw RTMLException if the class does not exist
     */
    virtual int train(Label classLabel)
    {
        int nbIterations = ConcurrentModels<ModelType, phraseType>::train(classLabel);
        updateTransitionParameters();
        return nbIterations;
    }
    
    virtual map<Label, int> retrain()
    {
        map<Label, int> nbIterations= ConcurrentModels<ModelType, phraseType>::retrain();
        updateTransitionParameters();
        return nbIterations;
    }
    
    virtual map<Label, int> train()
    {
        map<Label, int> nbIterations= ConcurrentModels<ModelType, phraseType>::train();
        updateTransitionParameters();
        return nbIterations;
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    /*!
     Write to JSON Node
     */
    using ConcurrentModels<ModelType, phraseType>::to_json;
    virtual JSONNode to_json() const
    {
        JSONNode json_hmodel(JSON_NODE);
        json_hmodel.set_name("Hierarchical Model");
        
        // Write Parent: Concurrent models
        JSONNode json_ccmodel = ConcurrentModels<ModelType, phraseType>::to_json();
        json_ccmodel.set_name("parent");
        json_hmodel.push_back(json_ccmodel);
        
        json_hmodel.push_back(JSONNode("incrementalLearning", incrementalLearning));
        
        // Write Prior
        JSONNode json_prior(JSON_ARRAY);
        json_prior.set_name("prior");
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; it++)
            json_prior.push_back(JSONNode("", prior.at(it->first)));
        json_hmodel.push_back(json_prior);
        
        // Write Exit Probabilities
        JSONNode json_exit(JSON_ARRAY);
        json_exit.set_name("exit");
        for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; it++)
            json_exit.push_back(JSONNode("", exitTransition.at(it->first)));
        json_hmodel.push_back(json_exit);
        
        // Write Transition Matrix
        JSONNode json_transition(JSON_ARRAY);
        json_transition.set_name("transition");
        for (const_model_iterator it1 = this->models.begin() ; it1 != this->models.end() ; it1++)
            for (const_model_iterator it2 = this->models.begin() ; it2 != this->models.end() ; it2++)
                json_transition.push_back(JSONNode("", transition.at(it1->first).at(it2->first)));
        json_hmodel.push_back(json_transition);
        
        return json_hmodel;
    }
    
    /*!
     Read from JSON Node
     */
    virtual void from_json(JSONNode root)
    {
        try {
            assert(root.type() == JSON_NODE);
            JSONNode::iterator root_it = root.begin();
            
            // Get Parent: Concurrent models
            assert(root_it != root.end());
            assert(root_it->name() == "parent");
            assert(root_it->type() == JSON_NODE);
            ConcurrentModels<ModelType, phraseType>::from_json(*root_it);
            root_it++;
            
            // Get Learning Model
            assert(root_it != root.end());
            assert(root_it->name() == "incrementalLearning");
            assert(root_it->type() == JSON_BOOL);
            incrementalLearning = root_it->as_bool();
            root_it++;
            
            // Get Prior
            assert(root_it != root.end());
            assert(root_it->name() == "prior");
            assert(root_it->type() == JSON_ARRAY);
            prior.clear();
            JSONNode::const_iterator array_it = (*root_it).begin();
            for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; it++) {
                assert(array_it != root.end());
                prior[it->first] = double(array_it->as_float());
                array_it++;
            }
            root_it++;
            
            // Get Exit Probabilities
            assert(root_it != root.end());
            assert(root_it->name() == "exit");
            assert(root_it->type() == JSON_ARRAY);
            exitTransition.clear();
            array_it = (*root_it).begin();
            for (const_model_iterator it = this->models.begin() ; it != this->models.end() ; it++) {
                assert(array_it != root.end());
                exitTransition[it->first] = double(array_it->as_float());
                array_it++;
            }
            root_it++;
            
            // Get Prior
            assert(root_it != root.end());
            assert(root_it->name() == "transition");
            assert(root_it->type() == JSON_ARRAY);
            transition.clear();
            array_it = (*root_it).begin();
            for (const_model_iterator it1 = this->models.begin() ; it1 != this->models.end() ; it1++) {
                for (const_model_iterator it2 = this->models.begin() ; it2 != this->models.end() ; it2++)
                {
                    assert(array_it != root.end());
                    transition[it1->first][it2->first] = double(array_it->as_float());
                    array_it++;
                }
            }
            
        } catch (exception &e) {
            throw RTMLException("Error reading JSON, Node: " + root.name());
        }
    }
    
    virtual void write(ostream& outStream)
    {
        outStream << "# =============================================================\n";
        outStream << "# =============================================================\n";
        outStream << "# HIERARCHICAL MODEL\n";
        outStream << "# =============================================================\n";
        outStream << "# =============================================================\n";
        outStream << "# incremental learning\n";
        outStream << incrementalLearning << endl;
        // TODO: Write high level transition parameters
        ConcurrentModels<ModelType, phraseType>::write(outStream);
    }
    
    virtual void read(istream& inStream)
    {
        //TODO: read something maybe?
        ConcurrentModels<ModelType, phraseType>::read(inStream);
    }
#pragma mark -
#pragma mark Python
    /*! @name Python methods */
#ifdef SWIGPYTHON
    void set_prior(int nbPrimitives, double *prior_) {
        if (nbPrimitives != this->size())
            throw RTMLException("Prior vector: wrong size", __FILE__, __FUNCTION__, __LINE__);
        this->set_prior(prior_);
    }
    
    void set_transition(int nbPrimitivesSquared, double *trans_) {
        if (nbPrimitivesSquared != this->size()*this->size())
            throw RTMLException("Transition matrix: wrong size", __FILE__, __FUNCTION__, __LINE__);
        this->set_transition(trans_);
    }
#endif
    
#pragma mark -
#pragma mark Protected attributes
protected:
    bool incrementalLearning; //!< learning mode: incremental learning if true, else: ergodic transitions
};

#endif
