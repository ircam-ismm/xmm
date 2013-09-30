//
//  concurrent_hmm.h
//  mhmm
//
//  Created by Jules Francoise on 08/08/13.
//
//

#ifndef mhmm_concurrent_hmm_h
#define mhmm_concurrent_hmm_h

#include "concurrent_models.h"
#include "hmm.h"

#pragma mark -
#pragma mark Class Definition
template<bool ownData>
class ConcurrentHMM : public ConcurrentModels<HMM<ownData>, Phrase<ownData, 1>, int> {
public:
    typedef typename  map<int, HMM<ownData> >::iterator model_iterator;
    typedef typename  map<int, HMM<ownData> >::const_iterator const_model_iterator;
    typedef typename  map<int, int>::iterator labels_iterator;
    
    bool updateWithEstimatedObservation;
    
    ConcurrentHMM(TrainingSet<Phrase<ownData, 1>, int> *_globalTrainingSet=NULL)
    : ConcurrentModels<HMM<ownData>, Phrase<ownData, 1>, int>(_globalTrainingSet)
    {
    }
    
#pragma mark -
#pragma mark Get & Set
    int get_dimension()
    {
        return this->referenceModel.get_dimension();
    }
    
    int get_nbStates()
    {
        return this->referenceModel.get_nbStates();
    }
    
    void set_nbStates(int nbStates_)
    {
        this->referenceModel.set_nbStates(nbStates_);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_nbStates(nbStates_);
        }
    }
    
    string get_transitionMode()
    {
        return this->referenceModel.get_transitionMode();
    }
    
    void set_transitionMode(string transMode_str)
    {
        this->referenceModel.set_transitionMode(transMode_str);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_transitionMode(transMode_str);
        }
    }
    
    int get_nbMixtureComponents() const
    {
        return this->referenceModel.get_nbMixtureComponents();
    }
    
    void set_nbMixtureComponents(int nbMixtureComponents_)
    {
        this->referenceModel.set_nbMixtureComponents(nbMixtureComponents_);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_nbMixtureComponents(nbMixtureComponents_);
        }
    }
    
    float  get_covarianceOffset()
    {
        return this->referenceModel.get_covarianceOffset();
    }
    
    void   set_covarianceOffset(float covarianceOffset_)
    {
        this->referenceModel.set_covarianceOffset(covarianceOffset_);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_covarianceOffset(covarianceOffset_);
        }
    }
    
    string get_EM_stopCriterion()
    {
        return this->referenceModel.get_EM_stopCriterion();
    }
    
    int get_EM_steps()
    {
        return this->referenceModel.get_EM_steps();
    }
    
    double get_EM_maxLogLikPercentChg()
    {
        return this->referenceModel.get_EM_maxLogLikPercentChg();
    }
    
    void set_EM_stopCriterion(string criterion)
    {
        this->referenceModel.set_EM_stopCriterion(criterion);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_EM_stopCriterion(criterion);
        }
    }
    
    void set_EM_steps(int steps)
    {
        this->referenceModel.set_EM_steps(steps);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_EM_steps(steps);
        }
    }
    
    void set_EM_maxLogLikelihoodPercentChg(double logLikPercentChg_)
    {
        this->referenceModel.set_EM_maxLogLikelihoodPercentChg(logLikPercentChg_);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_EM_maxLogLikelihoodPercentChg(logLikPercentChg_);
        }
    }
    
    int get_likelihoodBufferSize() const
    {
        return this->referenceModel.get_likelihoodBufferSize();
    }
    
    void set_likelihoodBufferSize(int likelihoodBufferSize_)
    {
        this->referenceModel.set_likelihoodBufferSize(likelihoodBufferSize_);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_likelihoodBufferSize(likelihoodBufferSize_);
        }
    }
    
    bool get_estimateMeans() const
    {
        return this->referenceModel.estimateMeans;
    }
    
    void set_estimateMeans(bool _estimateMeans)
    {
        this->referenceModel.estimateMeans = _estimateMeans;
        for (model_iterator it = this->models.begin() ; it != this->models.end() ; it++)
            it->second.estimateMeans = _estimateMeans;
    }
    
#pragma mark -
#pragma mark Training
    virtual void finishTraining()
    {
        for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
            int nbStates = this->get_nbStates();
            it->second.transition[(nbStates-1)*nbStates] = 0.01; // Add Cyclic Transition probability
        }
    }
    
    virtual void initPlaying()
    {
        for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
            it->second.initPlaying();
        }
    }
    
#pragma mark -
#pragma mark Performance
    void addCyclicTransition(double proba)
    {
        for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
            it->second.addCyclicTransition(proba);
        }
    }
    
    virtual void play(float *obs, double *modelLikelihoods)
    {
        double norm_const(0.0);
        int i(0);
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++) {
            modelLikelihoods[i] = it->second.play(obs);
            norm_const += modelLikelihoods[i++];
        }
        
        for (unsigned int i=0; i<this->models.size(); i++)
            modelLikelihoods[i] /= norm_const;
    }
    
    HMMResults getResults(int classLabel)
    {
        if (this->models.find(classLabel) == this->models.end())
            throw RTMLException("Class Label Does not exist", __FILE__, __FUNCTION__, __LINE__);
        return this->models[classLabel].results;
    }
    
#pragma mark -
#pragma mark Python
#ifdef SWIGPYTHON
    void play(int dimension_, double *observation,
              int nbModels_, double *likelihoods,
              int nbModels__, double *cumulativelikelihoods)
    {
        int dimension = this->referenceModel.get_dimension();
        
        float *obs = new float[dimension];
        for (int d=0; d<dimension; d++) {
            obs[d] = float(observation[d]);
        }
        
        play(obs, likelihoods);
        
        int m(0);
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            cumulativelikelihoods[m++] = it->second.cumulativeloglikelihood;
        
        delete[] obs;
    }
#endif
};


#endif
