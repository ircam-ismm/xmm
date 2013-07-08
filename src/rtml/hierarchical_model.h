//
//  hierarchical_model.h
//  mhmm
//
//  Created by Jules Francoise on 18/06/13.
//
//

#ifndef mhmm_hierarchical_model_h
#define mhmm_hierarchical_model_h

#include "concurrent_models.h"

#define HMHMM_DEFAULT_EXITTRANSITION 0.1
#define HMHMM_DEFAULT_INCREMENTALLEARNING false

using namespace std;

#pragma mark -
#pragma mark Class Definition
/*!
 @class HierarchicalModel
 @brief Handle Hierarchical machine learning models (specific to HMMs)
 @todo class description
 @tparam modelType type of the models
 @tparam phraseType type of the phrase in the training set (@see Phrase, MultimodalPhrase, GestureSoundPhrase)
 @tparam labelType type of the labels for each class.
 */
template<typename ModelType, typename phraseType, typename labelType=int>
class HierarchicalModel
: public ConcurrentModels<ModelType, phraseType, labelType> {
public:
    /*! @name iterators */
    typedef typename  map<labelType, ModelType>::iterator model_iterator;
    typedef typename  map<labelType, ModelType>::const_iterator const_model_iterator;
    typedef typename  map<int, labelType>::iterator labels_iterator;
    typedef typename  set<labelType>::iterator labset_iterator;
    
    map<labelType, double> prior;
    map<labelType, double> exitTransition;
    map<labelType, map<labelType, double> > transition;
    
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors */
    HierarchicalModel(TrainingSet<phraseType, labelType> *_globalTrainingSet=NULL)
    : ConcurrentModels<ModelType, phraseType, labelType>(_globalTrainingSet)
    {
        incrementalLearning = HMHMM_DEFAULT_INCREMENTALLEARNING;
    }
    
    ~HierarchicalModel()
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
        for (labset_iterator srcit = this->globalTrainingSet->allLabels.begin() ; srcit != this->globalTrainingSet->allLabels.end() ; srcit++)
        {
            sumPrior += prior[*srcit];
            double sumTrans(0.0);
            for (labset_iterator dstit = this->globalTrainingSet->allLabels.begin() ; dstit != this->globalTrainingSet->allLabels.end() ; dstit++)
                sumTrans += transition[*srcit][*dstit];
            for (labset_iterator dstit = this->globalTrainingSet->allLabels.begin() ; dstit != this->globalTrainingSet->allLabels.end() ; dstit++)
                transition[*srcit][*dstit] /= sumTrans;
        }
        for (labset_iterator it = this->globalTrainingSet->allLabels.begin() ; it != this->globalTrainingSet->allLabels.end() ; it++)
            prior[*it] /= sumPrior;
    }
    
    /*!
     * @brief set a particular value of the transition matrix
     *
     * set trans(i,j) = proba
     * @param srcSegmentLabel origin segment
     * @param dstSegmentLabel target segment
     * @param proba probability of making a transition from srcSegmentLabel to dstSegmentLabel
     */
    void setOneTransition(labelType srcSegmentLabel, labelType dstSegmentLabel, double proba)
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
            for (labset_iterator it = this->globalTrainingSet->allLabels.begin() ; it != this->globalTrainingSet->allLabels.end() ; it++)
                if (prior.find(*it) == prior.end())
                {
                    prior[*it] += double(regularizationFactor);
                    prior[*it] /= double(nbPrimitives + regularizationFactor) ;
                } else {
                    prior[*it] = 1. / double(nbPrimitives + regularizationFactor);
                }
        } else {
            for (labset_iterator it = this->globalTrainingSet->allLabels.begin() ; it != this->globalTrainingSet->allLabels.end() ; it++)
                prior[*it] = 1. / double(nbPrimitives);
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
            map<labelType, map<labelType, double> > oldTransition = transition;;
            
            for (labset_iterator srcit = this->globalTrainingSet->allLabels.begin() ; srcit != this->globalTrainingSet->allLabels.end() ; srcit++)
            {
                for (labset_iterator dstit = this->globalTrainingSet->allLabels.begin() ; dstit != this->globalTrainingSet->allLabels.end() ; dstit++)
                {
                    if (transition.find(*srcit) == transition.end() || transition[*srcit].find(*dstit) == transition[*srcit].end())
                    {
                        transition[*srcit][*dstit] = 1/double(nbPrimitives+regularizationFactor);
                    } else {
                        transition[*srcit][*dstit] += double(regularizationFactor);
                        transition[*srcit][*dstit] /= double(nbPrimitives+regularizationFactor);
                    }
                }
                
                if (exitTransition.find(*srcit) == exitTransition.end())
                {
                    exitTransition[*srcit] = 1/double(nbPrimitives+regularizationFactor);
                } else {
                    exitTransition[*srcit] += double(regularizationFactor);
                    exitTransition[*srcit] /= double(nbPrimitives+regularizationFactor);
                }
            }
        } else {
            for (labset_iterator srcit = this->globalTrainingSet->allLabels.begin() ; srcit != this->globalTrainingSet->allLabels.end() ; srcit++)
            {
                exitTransition[*srcit] = HMHMM_DEFAULT_EXITTRANSITION;
                
                for (labset_iterator dstit = this->globalTrainingSet->allLabels.begin() ; dstit != this->globalTrainingSet->allLabels.end() ; dstit++)
                    transition[*srcit][*dstit] = 1/(double)nbPrimitives;
            }
        }
    }
    
    /*!
     * @brief ergodic learning update high-level prior probabilities -> equal prior probs
     */
    void updatePrior_ergodic()
    {
        int nbPrimitives = this->size();
        for (labset_iterator it = this->globalTrainingSet->allLabels.begin() ; it != this->globalTrainingSet->allLabels.end() ; it++)
            prior[*it] = 1/double(nbPrimitives);
    }
    
    /*!
     * @brief ergodic learning: update high-level transition matrix
     *
     * (equal transition probabilities between primitive gestures)
     */
    void updateTransition_ergodic()
    {
        int nbPrimitives = this->size();
        for (labset_iterator srcit = this->globalTrainingSet->allLabels.begin() ; srcit != this->globalTrainingSet->allLabels.end() ; srcit++)
        {
            exitTransition[*srcit] = HMHMM_DEFAULT_EXITTRANSITION;
            for (labset_iterator dstit = this->globalTrainingSet->allLabels.begin() ; dstit != this->globalTrainingSet->allLabels.end() ; dstit++)
                transition[*srcit][*dstit] =  1/double(nbPrimitives);
        }
    }
    
    void updateExitProbabilities()
    {
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.updateExitProbabilities();
        }
    }
    
#pragma mark -
#pragma mark Training
    /*! @name Training */
    /*!
     Initialize training for each model
     */
    virtual void initTraining()
    {
        ConcurrentModels<ModelType, phraseType, labelType>::initTraining();
        updateTransitionParameters();
    }
    
    virtual map<labelType, int> retrain()
    {
        map<labelType, int> nbIterations= ConcurrentModels<ModelType, phraseType, labelType>::retrain();
        updateTransitionParameters();
        return nbIterations;
    }
    
#pragma mark -
#pragma mark Playing
    /*! @name Playing */
    virtual void initPlaying()
    {
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.initPlaying();
        }
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    virtual void write(ostream& outStream, bool writeTrainingSet=false)
    {
        outStream << "# =============================================================\n";
        outStream << "# =============================================================\n";
        outStream << "# HIERARCHICAL MODEL\n";
        outStream << "# =============================================================\n";
        outStream << "# =============================================================\n";
        outStream << "# incremental learning\n";
        outStream << incrementalLearning << endl;
        // TODO: Write high level transition parameters
        ConcurrentModels<ModelType, phraseType, labelType>::write(outStream, writeTrainingSet);
    }
    
    virtual void read(istream& inStream, bool readTrainingSet=false)
    {
        //TODO: read something maybe?
        ConcurrentModels<ModelType, phraseType, labelType>::read(inStream, readTrainingSet);
    }
#pragma mark -
#pragma mark Python
    /*! @name Python methods */
#ifdef SWIGPYTHON
    void set_prior(int nbPrimitives, double *prior_) {
        this->set_prior(prior_);
    }
    
    void set_transition(int nbPrimitivesSquared, double *trans_) {
        this->set_transition(trans_);
    }
#endif
    
#pragma mark -
#pragma mark Protected attributes
protected:
    bool incrementalLearning; //!< learning mode: incremental learning if true, else: ergodic transitions
};

#endif
