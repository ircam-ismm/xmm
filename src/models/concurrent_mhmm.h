//
//  concurrent_mhmm.h
//  mhmm
//
//  Created by Jules Francoise on 29/01/13.
//
//

#ifndef mhmm_concurrent_mhmm_h
#define mhmm_concurrent_mhmm_h

#include "concurrent_models.h"
#include "multimodal_hmm.h"

#pragma mark -
#pragma mark Class Definition
template<bool ownData>
class ConcurrentMHMM : public ConcurrentModels< MultimodalHMM<ownData>, GestureSoundPhrase<ownData> > {
public:
    typedef typename  map<Label, MultimodalHMM<ownData> >::iterator model_iterator;
    typedef typename  map<Label, MultimodalHMM<ownData> >::const_iterator const_model_iterator;
    typedef typename  map<int, Label>::iterator labels_iterator;
    
    bool updateWithEstimatedObservation;
    
    ConcurrentMHMM(TrainingSet< GestureSoundPhrase<ownData> > *_globalTrainingSet=NULL)
    : ConcurrentModels< MultimodalHMM<ownData>, GestureSoundPhrase<ownData> >(_globalTrainingSet)
    {
        updateWithEstimatedObservation = false;
    }
    
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
        int dimension_gesture = this->referenceModel.get_dimension_gesture();
        int dimension_sound = this->referenceModel.get_dimension_sound();
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
            
            double maxLikelihood;
            int i(0);
            
            // Parallel Version: Bugsssss
            /*
            vector< future<double> > modelLikelihoods_future(this->size());
            for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            {
                modelLikelihoods_future[i++] = async(launch::async, &MultimodalHMM<ownData>::play, &it->second, obs_ref);
            }
            
            for (i=0 ; i<this->size() ; i++)
            {
                modelLikelihoods[i] = modelLikelihoods_future[i].get();
                norm_const += modelLikelihoods[i];
            }
            
            i = 0;
            for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            {
                if (i==0 || modelLikelihoods[i] > maxLikelihood)
                {
                    maxLikelihood = modelLikelihoods[i];
                    copy(it->second.results.observation_sound.begin(),
                         it->second.results.observation_sound.end(),
                         obs+dimension_gesture);
                }
                i++;
            }
            //*/
            
            //*
            i = 0;
            model_iterator it = this->models.begin();
            maxLikelihood = it->second.play(obs_ref);
            modelLikelihoods[i++] = maxLikelihood;
            norm_const = maxLikelihood;
            copy(obs_ref+dimension_gesture, obs_ref+dimension_gesture+dimension_sound, obs+dimension_gesture);
            
            it++;
            while (it != this->models.end()) {
                double alphaLikelihood = it->second.play(obs_ref);
                modelLikelihoods[i] = alphaLikelihood;
                norm_const += alphaLikelihood;
                if (alphaLikelihood > maxLikelihood) {
                    maxLikelihood = alphaLikelihood;
                    copy(it->second.results.observation_sound.begin(),
                         it->second.results.observation_sound.end(),
                         obs+dimension_gesture);
                }
                it++;
                i++;
            }
            //*/
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
        
        // TODO: test this with multiple models
        if (updateWithEstimatedObservation) {
            int i(0);
            for (model_iterator it = this->models.begin(); it != this->models.end() ; it++) {
                modelLikelihoods[i] = it->second.forward_update_withNewObservation(obs, obs + dimension_gesture);
                norm_const += modelLikelihoods[i++];
            }
        }
        
        for (unsigned int i=0; i<this->models.size(); i++)
            modelLikelihoods[i] /= norm_const;
        
        // Invert and Normalize covariance determinants
        /*
        double s(0.0);
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            s += 1./it->second.results.covarianceDeterminant_sound;
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            it->second.results.covarianceDeterminant_sound = (1./it->second.results.covarianceDeterminant_sound) / s;
        //*/
        
        delete[] obs_ref;
    }
    
    MHMMResults getResults(Label classLabel)
    {
        if (this->models.find(classLabel) == this->models.end())
            throw RTMLException("Class Label Does not exist", __FILE__, __FUNCTION__, __LINE__);
        return this->models[classLabel].results;
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
