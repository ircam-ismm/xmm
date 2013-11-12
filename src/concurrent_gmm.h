//
//  concurrent_gmm.h
//  mhmm
//
//  Created by Jules Francoise on 08/08/13.
//
//

#ifndef mhmm_concurrent_gmm_h
#define mhmm_concurrent_gmm_h

#include "concurrent_models.h"
#include "gmm.h"

#pragma mark -
#pragma mark Class Definition
template<bool ownData>
class ConcurrentGMM : public ConcurrentModels<GMM<ownData>, Phrase<ownData, 1>, int> {
public:
    typedef typename  map<int, GMM<ownData> >::iterator model_iterator;
    typedef typename  map<int, GMM<ownData> >::const_iterator const_model_iterator;
    typedef typename  map<int, int>::iterator labels_iterator;
    
    ConcurrentGMM(TrainingSet<Phrase<ownData, 1>, int> *_globalTrainingSet=NULL)
    : ConcurrentModels<GMM<ownData>, Phrase<ownData, 1>, int>(_globalTrainingSet)
    {}
    
#pragma mark -
#pragma mark Get & Set
    int get_dimension()
    {
        return this->referenceModel.get_dimension();
    }
    
    int get_nbMixtureComponents()
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
    
    int get_EM_minSteps()
    {
        return this->referenceModel.get_EM_minSteps();
    }
    
    int get_EM_maxSteps()
    {
        return this->referenceModel.get_EM_maxSteps();
    }
    
    double get_EM_percentChange()
    {
        return this->referenceModel.get_EM_percentChange();
    }
    
    void set_EM_minSteps(int steps)
    {
        this->referenceModel.set_EM_minSteps(steps);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_EM_minSteps(steps);
        }
    }
    
    void set_EM_maxSteps(int steps)
    {
        this->referenceModel.set_EM_maxSteps(steps);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_EM_maxSteps(steps);
        }
    }
    
    void set_EM_percentChange(double logLikPercentChg_)
    {
        this->referenceModel.set_EM_percentChange(logLikPercentChg_);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_EM_percentChange(logLikPercentChg_);
        }
    }
    
    unsigned int get_likelihoodBufferSize() const
    {
        return this->referenceModel.get_likelihoodBufferSize();
    }
    
    void set_likelihoodBufferSize(unsigned int likelihoodBufferSize_)
    {
        this->referenceModel.set_likelihoodBufferSize(likelihoodBufferSize_);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.set_likelihoodBufferSize(likelihoodBufferSize_);
        }
    }
    
#pragma mark -
#pragma mark Playing
    void initPlaying()
    {
        for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
            it->second.initPlaying();
        }
    }
    
    void play(float *obs, double *modelLikelihoods)
    {
        double norm_const(0.0);
        int i(0);
        for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
            modelLikelihoods[i] = it->second.play(obs);
            norm_const += modelLikelihoods[i++];
        }
        
        for (unsigned int i=0; i<this->models.size(); i++)
            modelLikelihoods[i] /= norm_const;
    }
    
#pragma mark -
#pragma mark Python
#ifdef SWIGPYTHON
    void play(int dimension_, double *observation,
              int nbModels_, double *likelihoods,
              int nbModels__, double *cumulativelikelihoods)
    {
        int dimension = this->referenceModel.get_dimension();
        
        float *obs_float = new float[dimension];
        for (int i=0 ; i<dimension ; i++)
            obs_float[i] = float(observation[i]);
        
        play(obs_float, likelihoods);
        
        int m(0);
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            cumulativelikelihoods[m++] = it->second.cumulativeloglikelihood;
        
        delete[] obs_float;
    }
#endif
};


#endif
