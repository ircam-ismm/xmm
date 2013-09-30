//
//  concurrent_mgmm.h
//  mhmm
//
//  Created by Jules Francoise on 28/01/13.
//
//

#ifndef mhmm_concurrent_mgmm_h
#define mhmm_concurrent_mgmm_h

#include "concurrent_models.h"
#include "multimodal_gmm.h"

#pragma mark -
#pragma mark Class Definition
template<bool ownData>
class ConcurrentMGMM : public ConcurrentModels<MultimodalGMM<ownData>, GestureSoundPhrase<ownData>, int> {
public:
    typedef typename  map<int, MultimodalGMM<ownData> >::iterator model_iterator;
    typedef typename  map<int, MultimodalGMM<ownData> >::const_iterator const_model_iterator;
    typedef typename  map<int, int>::iterator labels_iterator;
    
    ConcurrentMGMM(TrainingSet<GestureSoundPhrase<ownData>, int> *_globalTrainingSet=NULL)
    : ConcurrentModels<MultimodalGMM<ownData>, GestureSoundPhrase<ownData>, int>(_globalTrainingSet)
    {}

#pragma mark -
#pragma mark Get & Set
    int get_dimension_gesture()
    {
        return this->referenceModel.get_dimension_gesture();
    }
    
    int get_dimension_sound()
    {
        return this->referenceModel.get_dimension_sound();
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
        int dimension_gesture = this->get_dimension_gesture();
        int dimension_sound = this->get_dimension_sound();
        int dimension_total = dimension_gesture + dimension_sound;
        
        float* obs_ref = new float[dimension_total];
        copy(obs, obs+dimension_total, obs_ref);
        
        for (int d=0; d<dimension_sound; d++) {
            obs[dimension_gesture+d] = 0.0;
        }
        
        double norm_const(0.0);
        if (this->playMode == this->LIKELIEST)
        {
            // EVALUATE SOUND OBSERVATIONS ON LIKELIEST MODEL
            int i(0);
            model_iterator it = this->models.begin();
            double maxLikelihood = it->second.play(obs_ref);
            modelLikelihoods[i++] = maxLikelihood;
            norm_const = maxLikelihood;
            copy(obs_ref+dimension_gesture, obs_ref+dimension_gesture+dimension_sound, obs+dimension_gesture);
            
            it++;
            while (it != this->models.end()) {
                double alphaLikelihood = it->second.play(obs_ref);
                modelLikelihoods[i++] = alphaLikelihood;
                norm_const += alphaLikelihood;
                if (alphaLikelihood > maxLikelihood) {
                    copy(obs_ref+dimension_gesture, obs_ref+dimension_gesture+dimension_sound, obs+dimension_gesture);
                    maxLikelihood = alphaLikelihood;
                }
                it++;
            }
        }
        else // polyPlayMode == MIXTURE
        {
            // EVALUATE SOUND OBSERVATIONS AS A MIXTURE OF MODELS
            int i(0);
            for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
                double alphaLikelihood = it->second.play(obs_ref);
                norm_const += alphaLikelihood;
                modelLikelihoods[i++] = alphaLikelihood;
                for (int d=0; d<dimension_sound; d++) {
                    obs[dimension_gesture+d] += alphaLikelihood * obs_ref[dimension_gesture+d];
                }
            }
            
            for (int d=0; d<dimension_sound; d++) {
                obs[dimension_gesture+d] /= norm_const;
            }
        }
        
        for (unsigned int i=0; i<this->models.size(); i++)
            modelLikelihoods[i] /= norm_const;
        
        // Invert and Normalize covariance determinants
        double s(0.0);
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            s += 1./it->second.covariance_sound_det;
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            it->second.covariance_sound_det = (1./it->second.covariance_sound_det) / s;
        
        delete[] obs_ref;
    }
    
#pragma mark -
#pragma mark Python
#ifdef SWIGPYTHON
    void play(int dimension_gesture_, double *observation_gesture,
              int dimension_sound_, double *observation_sound_out,
              int nbModels_, double *likelihoods,
              int nbModels__, double *cumulativelikelihoods)
    {
        int dimension_gesture = this->referenceModel.get_dimension_gesture();
        int dimension_sound = this->referenceModel.get_dimension_sound();
        int dimension_total = dimension_gesture + dimension_sound;
        
        float *observation_total = new float[dimension_total];
        for (int d=0; d<dimension_gesture; d++) {
            observation_total[d] = float(observation_gesture[d]);
        }
        for (int d=0; d<dimension_sound; d++)
            observation_total[d+dimension_gesture] = 0.;
        
        play(observation_total, likelihoods);
        
        for (int d=0; d<dimension_sound_; d++) {
            observation_sound_out[d] = double(observation_total[dimension_gesture+d]);
        }
        
        int m(0);
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            cumulativelikelihoods[m++] = it->second.cumulativeloglikelihood;
        
        delete[] observation_total;
    }
#endif
};

#endif
