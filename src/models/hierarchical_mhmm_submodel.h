//
//  hierarchical_mhmm_submodel.h
//  mhmm
//
//  Created by Jules Francoise on 18/06/13.
//
//

#ifndef mhmm_hierarchical_mhmm_submodel_h
#define mhmm_hierarchical_mhmm_submodel_h

#include "multimodal_hmm.h"

#define HMHMM_DEFAULT_EXITPROBABILITY 0.1

using namespace std;

struct HMHMMResults : public MHMMResults {
    double exitLikelihood;
    double likelihoodnorm;
};

template<bool ownData> class HierarchicalMHMM;

template<bool ownData>
class HierarchicalMHMMSubmodel : public MultimodalHMM<ownData>
{
    friend class HierarchicalMHMM<ownData>;
public:
    HMHMMResults results_h;
    
    // Forward variables
	vector<double> alpha_h[3] ; //!< Alpha variable (forward algorithm)
    
#pragma mark -
#pragma mark Constructors
    HierarchicalMHMMSubmodel(TrainingSet< GestureSoundPhrase<ownData> > *_trainingSet=NULL,
                             int nbStates_ = MHMM_DEFAULT_NB_STATES,
                             int nbMixtureComponents_ = MGMM_DEFAULT_NB_MIXTURE_COMPONENTS,
                             float covarianceOffset_ = MGMM_DEFAULT_COVARIANCE_OFFSET)
    : MultimodalHMM<ownData>(_trainingSet, nbStates_, nbMixtureComponents_, covarianceOffset_)
    {
        updateExitProbabilities(NULL);
    }
    
    virtual ~HierarchicalMHMMSubmodel()
    {
        for (int i=0 ; i<3 ; i++)
            this->alpha_h[i].clear();
        this->exitProbabilities.clear();
    }
    
#pragma mark -
#pragma mark Exit Probabilities
    void updateExitProbabilities(float *_exitProbabilities = NULL)
    {
        if (_exitProbabilities == NULL) {
            exitProbabilities.resize(this->nbStates, 0.0);
            exitProbabilities[this->nbStates-1] = HMHMM_DEFAULT_EXITPROBABILITY;
        } else {
            exitProbabilities.resize(this->nbStates, 0.0);
            for (int i=0 ; i < this->nbStates ; i++)
                try {
                    exitProbabilities[i] = _exitProbabilities[i];
                } catch (exception &e) {
                    throw RTMLException("Error reading exit probabilities", __FILE__, __FUNCTION__, __LINE__);
                }
        }
    }
    
    void addExitPoint(int state, float proba)
    {
        if (state >= this->nbStates)
            throw RTMLException("State index out of bounds", __FILE__, __FUNCTION__, __LINE__);
        exitProbabilities[state] = proba;
    }

#pragma mark -
#pragma mark Play !
    void initPlaying()
    {
        MultimodalHMM<ownData>::initPlaying();
        for (int i=0 ; i<3 ; i++)
            alpha_h[i].resize(this->nbStates, 0.0);
    }
    
    virtual void estimateObservation(float *obs)
    {
        float *obs2 = new float[this->dimension_total];
        copy(obs, obs+this->dimension_total, obs2);
        for (int d=0; d<this->dimension_sound; d++) {
            obs[this->dimension_gesture+d] = 0.0;
        }
        for (int i=0; i<this->nbStates; i++) {
            this->states[i].regression(obs2);
            for (int d=0; d<this->dimension_sound; d++) {
                obs[this->dimension_gesture+d] += (alpha_h[0][i] + alpha_h[1][i]) * obs2[this->dimension_gesture+d];
            }
        }
    }

#pragma mark -
#pragma mark protected Attributes
protected:
    vector<float> exitProbabilities;
};

#endif
