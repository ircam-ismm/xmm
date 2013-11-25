//
// em_based_learning_model.h
//
// Machine learning model based on the EM algorithm
//
// Copyright (C) 2013 Ircam - Jules Françoise. All Rights Reserved.
// author: Jules Françoise
// contact: jules.francoise@ircam.fr
//

#ifndef rtml_em_based_learning_model_h
#define rtml_em_based_learning_model_h

#include "learning_model.h"
#include <cmath>

using namespace std;

const int EM_MODEL_DEFAULT_EMSTOP_MINSTEPS = 10;
const int EM_MODEL_DEFAULT_EMSTOP_MAXSTEPS = 0;
const double EM_MODEL_DEFAULT_EMSTOP_PERCENT_CHG = 0.01;
const int EM_MODEL_DEFAULT_LIKELIHOOD_WINDOW = 1;

/*!
 @brief Stop Criterion for the EM algorithm
 */
struct EMStopCriterion {
    int    minSteps;                //!< number of EM iterations is STEPS EM criterion
    int    maxSteps;                //!< number of EM iterations is STEPS EM criterion
    double percentChg;           //!< log-likelihood difference threshold to stop EM re-estimation
};

#pragma mark -
#pragma mark Class Definition
/*!
 @class EMBasedLearningModel
 @brief Generic Template for Machine Learning Probabilistic models based on the EM algorithm
 @tparam phraseType type of the phrase in the training set (@see Phrase, MultimodalPhrase, GestureSoundPhrase)
 */
template <typename phraseType>
class EMBasedLearningModel : public LearningModel<phraseType>
{
public:
    double cumulativeloglikelihood;
    
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors*/
    /*!
     Constructor
     @param _trainingSet training set associated with the model
     */
    EMBasedLearningModel(TrainingSet<phraseType> *_trainingSet)
    : LearningModel<phraseType>(_trainingSet)
    {
        stopcriterion.minSteps = EM_MODEL_DEFAULT_EMSTOP_MINSTEPS;
        stopcriterion.maxSteps = EM_MODEL_DEFAULT_EMSTOP_MAXSTEPS;
        stopcriterion.percentChg = EM_MODEL_DEFAULT_EMSTOP_PERCENT_CHG;
        likelihoodBuffer.resize(EM_MODEL_DEFAULT_LIKELIHOOD_WINDOW);
    }
    
    /*!
     Copy Constructor
     */
    EMBasedLearningModel(EMBasedLearningModel<phraseType> const& src) : LearningModel<phraseType>(src)
    {
        this->_copy(this, src);
    }
    
    /*!
     Assignment
     */
    EMBasedLearningModel<phraseType>& operator=(EMBasedLearningModel<phraseType> const& src)
    {
        if(this != &src)
        {
            _copy(this, src);
        }
        return *this;
    };
    
    /*!
     Copy between two models
     */
    using LearningModel<phraseType>::_copy;
    virtual void _copy(EMBasedLearningModel<phraseType> *dst,
                       EMBasedLearningModel<phraseType> const& src)
    {
        LearningModel<phraseType>::_copy(dst, src);
        dst->stopcriterion.minSteps = src.stopcriterion.minSteps;
        dst->stopcriterion.maxSteps = src.stopcriterion.maxSteps;
        dst->stopcriterion.percentChg = src.stopcriterion.percentChg;
        dst->likelihoodBuffer.resize(src.likelihoodBuffer.size());
        dst->likelihoodBuffer.clear();
    }
    
    virtual ~EMBasedLearningModel()
    {}
    
#pragma mark -
#pragma mark Training
    /*! @name Training algorithm */
    /*!
     Main training method based on the EM algorithm\n
     the method performs a loop over the pure virtual method train_EM_update() until convergence.
     @see train_EM_update()
     */
    int train()
    {
        if (!this->trainingSet)
            throw RTMLException("No training Set is Connected", __FILE__, __FUNCTION__, __LINE__);
        
        this->initTraining();
        
        if (this->trainingSet->is_empty())
            throw RTMLException("No training data", __FILE__, __FUNCTION__, __LINE__);
        
        double log_prob(log(0.)), old_log_prob;
        int nbIterations(0);
        
        do {
            old_log_prob = log_prob;
            log_prob = train_EM_update();
            
            /*
             cout << "step "<< nbIterations
             << ": precent-change = " << 100.*fabs((log_prob-old_log_prob)/old_log_prob)
             << ", logProb = " << log_prob << endl;
             //*/
            
            nbIterations++;
            
            if (isnan(100.*fabs((log_prob-old_log_prob)/old_log_prob)) && (nbIterations > 1)) { //  (nbIterations > 0 && log_prob == 0.0)
                throw RTMLException("Training Error: No convergence! Try again... (maybe change nb of states or increase covarianceOffset)", __FILE__, __FUNCTION__, __LINE__);
            }
        } while (!train_EM_stop(nbIterations, log_prob, old_log_prob));
        
        this->finishTraining();
        this->trained = true;
        this->trainingSet->set_unchanged();
        return nbIterations;
    }
    
#pragma mark -
#pragma mark EM Stop Criterion
    /*!
     Get minimum number of EM steps
     */
    int get_EM_minSteps() const
    {
        return stopcriterion.minSteps;
    }
    
    /*!
     Get maximum number of EM steps
     */
    int get_EM_maxSteps() const
    {
        return stopcriterion.maxSteps;
    }
    
    /*!
     Get EM convergence threshold in percent-change of the likelihood
     */
    double get_EM_percentChange() const
    {
        return stopcriterion.percentChg;
    }
    
    /*!
     Set minimum number of steps of the EM algorithm
     */
    void set_EM_minSteps(int steps)
    {
        if (steps < 1) throw RTMLException("Minimum number of EM steps must be > 0", __FILE__, __FUNCTION__, __LINE__);
        
        stopcriterion.minSteps = steps;
    }
    
    /*!
     Set maximum number of steps of the EM algorithm
     */
    void set_EM_maxSteps(int steps)
    {
        if (steps < 1) throw RTMLException("Maximum number of EM steps must be > 0", __FILE__, __FUNCTION__, __LINE__);
        
        stopcriterion.maxSteps = steps;
    }
    
    /*!
     Set convergence threshold in percent-change of the likelihood
     */
    void set_EM_percentChange(double logLikelihoodPercentChg)
    {
        if (logLikelihoodPercentChg > 0) {
            stopcriterion.percentChg = logLikelihoodPercentChg;
        } else {
            throw RTMLException("Max loglikelihood difference for EM stop criterion must be > 0", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    
    /*!
     checks if the training has converged according to the object's EM stop criterion
     @param step current step of the EM algorithm
     @param log_prob log-likelihood returned by the EM update
     @param old_log_prob log-likelihood returned by the EM update at the previous step
     */
    bool train_EM_stop(int step, double log_prob, double old_log_prob) const
    {
        if (stopcriterion.maxSteps > stopcriterion.minSteps)
            return (step >= stopcriterion.maxSteps);
        else
            return (step >= stopcriterion.minSteps) && (100.*fabs((log_prob - old_log_prob) / log_prob) < stopcriterion.percentChg);
    }
    
#pragma mark -
#pragma mark Likelihood Buffer
    /*! @name Likelihood smoothing buffer */
    /*!
     get size of the likelihood smoothing buffer (number of frames)
     */
    int get_likelihoodBufferSize() const
    {
        return likelihoodBuffer.size();
    }
    
    /*!
     set size of the likelihood smoothing buffer (number of frames)
     */
    void set_likelihoodBufferSize(int likelihoodBufferSize_)
    {
        if (likelihoodBufferSize_ < 1) throw RTMLException("Likelihood Buffer size must be > 1", __FILE__, __FUNCTION__, __LINE__);
        likelihoodBuffer.resize(likelihoodBufferSize_);
    }
    
    /*!
     update the content of the likelihood buffer and return average likelihood.
     The method also updates the cumulative log-likelihood computed over a window (cumulativeloglikelihood)
     @param instantLikelihood instantaneous likelihood at the current step
     @return mean of the likelihood buffer
     */
    void updateLikelihoodBuffer(double instantLikelihood)
    {
        likelihoodBuffer.push(log(instantLikelihood));
        cumulativeloglikelihood = 0.0;
        unsigned int bufSize = likelihoodBuffer.size_t();
        for (unsigned int i=0; i<bufSize; i++) {
            cumulativeloglikelihood += likelihoodBuffer(0, i);
        }
        cumulativeloglikelihood /= double(bufSize);
    }
    
    /*!
     initialize the playing mode
     */
    virtual void initPlaying()
    {
        LearningModel<phraseType>::initPlaying();
        likelihoodBuffer.clear();
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    virtual void write(ostream& outStream, bool writeTrainingSet=false)
    {
        outStream << "# EM stop criterion\n";
        outStream << stopcriterion.minSteps << " " << stopcriterion.maxSteps << " " << stopcriterion.percentChg << endl;
        outStream << "# Size of the likehood buffer\n";
        outStream << likelihoodBuffer.size() << endl;
        LearningModel<phraseType>::write(outStream, writeTrainingSet);
    }
    
    virtual void read(istream& inStream, bool readTrainingSet=false)
    {
        // Get EM Stop Criterion
        skipComments(&inStream);
        inStream >> stopcriterion.minSteps;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        inStream >> stopcriterion.maxSteps;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        inStream >> stopcriterion.percentChg;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        // Size of the likehood buffer
        skipComments(&inStream);
        int _likelihoodBufferSize;
        inStream >> _likelihoodBufferSize;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        set_likelihoodBufferSize(_likelihoodBufferSize);
        LearningModel<phraseType>::read(inStream, readTrainingSet);
    }
    
#pragma mark -
#pragma mark Protected Attributes
    /*! @name Protected Attributes*/
protected:
    RingBuffer<double, 1> likelihoodBuffer;
    EMStopCriterion stopcriterion;
    
#pragma mark -
#pragma mark Pure virtual methods
    virtual double train_EM_update() = 0;
};

#endif