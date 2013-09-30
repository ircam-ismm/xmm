//
//  em_based_learning_model.h
//  rtml
//
//  Created by Jules Francoise on 21/01/13.
//
//

#ifndef rtml_em_based_learning_model_h
#define rtml_em_based_learning_model_h

#include "learning_model.h"
#include <cmath>

#define EM_MODEL_DEFAULT_EMSTOP_TYPE PERCENT_CHG
#define EM_MODEL_DEFAULT_EMSTOP_STEPS 10
#define EM_MODEL_DEFAULT_EMSTOP_PERCENT_CHG 0.01
#define EM_MODEL_DEFAULT_LIKELIHOOD_WINDOW 1

using namespace std;

/*!
 @enum EM_STOP_CRITERION_TYPE
 Type of stop criterion for the EM algorithm
 */
enum EM_STOP_CRITERION_TYPE {
    STEPS,       //!< Stop after a given number of steps
    PERCENT_CHG, //!< stop when the percent change of the log likelihood is under a given threshold
    BOTH         //!< Same as PERCENT_CHG with a minimum number of steps
};


/*!
 @brief Stop Criterion for the EM algorithm
 */
struct EMStopCriterion {
    EM_STOP_CRITERION_TYPE type; //!< type of criterion
    int    steps;                //!< number of EM iterations is STEPS EM criterion
    double percentChg;           //!< log-likelihood difference threshold to stop EM re-estimation
};

#pragma mark -
#pragma mark Class Definition
/*!
 @class EMBasedLearningModel
 @brief Generic Template for Machine Learning Probabilistic models based on the EM algorithm
 @tparam phraseType type of the phrase in the training set (@see Phrase, MultimodalPhrase, GestureSoundPhrase)
 @tparam labelType type of the labels for each class.
 */
template <typename phraseType, typename labelType=int>
class EMBasedLearningModel : public LearningModel<phraseType, labelType>
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
    EMBasedLearningModel(TrainingSet<phraseType, labelType> *_trainingSet)
    : LearningModel<phraseType, labelType>(_trainingSet)
    {
        stopcriterion.type = EM_MODEL_DEFAULT_EMSTOP_TYPE;
        stopcriterion.steps = EM_MODEL_DEFAULT_EMSTOP_STEPS;
        stopcriterion.percentChg = EM_MODEL_DEFAULT_EMSTOP_PERCENT_CHG;
        likelihoodBuffer.resize(EM_MODEL_DEFAULT_LIKELIHOOD_WINDOW);
    }
    
    /*!
     Copy Constructor
     */
    EMBasedLearningModel(EMBasedLearningModel<phraseType, labelType> const& src) : LearningModel<phraseType, labelType>(src)
    {
        this->_copy(this, src);
    }
    
    /*!
     Assignment
     */
    EMBasedLearningModel<phraseType, labelType>& operator=(EMBasedLearningModel<phraseType, labelType> const& src)
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
    virtual void _copy(EMBasedLearningModel<phraseType, labelType> *dst,
                       EMBasedLearningModel<phraseType, labelType> const& src)
    {
        LearningModel<phraseType, labelType>::_copy(dst, src);
        dst->stopcriterion.type = src.stopcriterion.type;
        dst->stopcriterion.steps = src.stopcriterion.steps;
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
    /*! @name EM algorithm stop criterion */
    /*!
     get the type of stop criterion of the EM algorithm
     @return string corresponding the type of stop criterion
     @see EMStopCriterion
     */
    string get_EM_stopCriterion() const
    {
        if (stopcriterion.type == STEPS)
            return "steps";
        if (stopcriterion.type == PERCENT_CHG)
            return "percentchg";
        return "both";
    }
    
    /*!
     Get number of steps for stop criterion
     */
    int get_EM_steps() const
    {
        return stopcriterion.steps;
    }
    
    /*!
     Get convergence threshold in percent-change of the likelihood
     */
    double get_EM_maxLogLikPercentChg() const
    {
        return stopcriterion.percentChg;
    }
    
    /*!
     set type of stop criterion of the EM algorithm
     @param criterion string corresponding to the type of criterion (steps / percentchg / both)
     */
    void set_EM_stopCriterion(string criterion)
    {
        if (criterion == "steps") {
            stopcriterion.type = STEPS;
        } else if (criterion == "percentchg") {
            stopcriterion.type = PERCENT_CHG;
        } else if (criterion == "both") {
            stopcriterion.type = BOTH;
        } else {
            throw RTMLException("Unknown EM Stop criterion", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    
    /*!
     Set number of steps of the EM algorithm
     */
    void set_EM_steps(int steps)
    {
        if (steps < 1) throw RTMLException("Max number of EM steps must be > 0", __FILE__, __FUNCTION__, __LINE__);
        
        stopcriterion.steps = steps;
    }
    
    /*!
     Set convergence threshold in percent-change of the likelihood
     */
    void set_EM_maxLogLikelihoodPercentChg(double logLikelihoodPercentChg)
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
        if (step > 120) return true;
        switch (stopcriterion.type) {
            case STEPS:
                return (step >= stopcriterion.steps);
            case PERCENT_CHG:
                return (100.*fabs((log_prob - old_log_prob) / log_prob) < stopcriterion.percentChg);
            default: //BOTH
                return (step >= stopcriterion.steps) && (100.*fabs((log_prob - old_log_prob) / log_prob) < stopcriterion.percentChg);
        }
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
    double updateLikelihoodBuffer(double instantLikelihood)
    {
        likelihoodBuffer.push(instantLikelihood);
        double mean(0.);
        cumulativeloglikelihood = 0.0;
        unsigned int bufSize = likelihoodBuffer.size_t();
        for (unsigned int i=0; i<bufSize; i++) {
            mean += likelihoodBuffer(0, i);
            cumulativeloglikelihood += log(likelihoodBuffer(0, i));
        }
        cumulativeloglikelihood /= double(bufSize);
        
        return mean / double(bufSize);
    }
    
    /*!
     initialize the playing mode
     */
    virtual void initPlaying()
    {
        likelihoodBuffer.clear();
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    virtual void write(ostream& outStream, bool writeTrainingSet=false)
    {
        outStream << "# EM stop criterion\n";
        outStream << stopcriterion.type << " " << stopcriterion.steps << " " << stopcriterion.percentChg << endl;
        outStream << "# Size of the likehood buffer\n";
        outStream << likelihoodBuffer.size() << endl;
        LearningModel<phraseType, labelType>::write(outStream, writeTrainingSet);
    }
    
    virtual void read(istream& inStream, bool readTrainingSet=false)
    {
        // Get EM Stop Criterion
        skipComments(&inStream);
        int sc_type;
        inStream >> sc_type;
        stopcriterion.type = EM_STOP_CRITERION_TYPE(sc_type);
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        inStream >> stopcriterion.steps;
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
        LearningModel<phraseType, labelType>::read(inStream, readTrainingSet);
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