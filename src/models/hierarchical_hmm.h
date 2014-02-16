//
//  hierarchical_hmm.h
//  mhmm
//
//  Created by Jules Francoise on 08/08/13.
//
//

#ifndef mhmm_hierarchical_hmm_h
#define mhmm_hierarchical_hmm_h

#include "hierarchical_model.h"
#include "hierarchical_hmm_submodel.h"

#define HHMM_DEFAULT_EXITTRANSITION 0.1

using namespace std;

#pragma mark -
#pragma mark Class Definition
template<bool ownData>
class HierarchicalHMM : public HierarchicalModel< HierarchicalHMMSubmodel<ownData>, Phrase<ownData, 1> > {
public:
    typedef typename  map<Label, HierarchicalHMMSubmodel<ownData> >::iterator model_iterator;
    typedef typename  map<Label, HierarchicalHMMSubmodel<ownData> >::const_iterator const_model_iterator;
    typedef typename  map<int, Label>::iterator labels_iterator;
    
    HierarchicalHMM(TrainingSet< Phrase<ownData, 1> > *_globalTrainingSet=NULL)
    : HierarchicalModel< HierarchicalHMMSubmodel<ownData>, Phrase<ownData, 1> >(_globalTrainingSet)
    {
        forwardInitialized = false;
    }
    
    virtual ~HierarchicalHMM()
    {
        V1.clear();
        V2.clear();
    }
    
#pragma mark -
#pragma mark Get & Set
    int get_dimension() const
    {
        return this->referenceModel.get_dimension();
    }
    
    int get_nbStates() const
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
    
    float  get_covarianceOffset() const
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
    
    void addExitPoint(int state, float proba)
    {
        this->referenceModel.addExitPoint(state, proba);
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.addExitPoint(state, proba);
        }
    }
    
#pragma mark -
#pragma mark Forward Algorithm
    model_iterator forward_init(float *observation, double *modelLikelihoods)
    {
        double norm_const(0.0) ;
        
        // TODO: is it useful to do it here? ==> better in play()
        // int dimension = this->get_dimension();
        
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++)
        {
            int N = it->second.get_nbStates();
            
            for (int i=0; i<3; i++) {
                for (int k=0; k<N; k++){
                    it->second.alpha_h[i][k] = 0.0;
                }
            }
            
            // Compute Emission probability and initialize on the first state of the primitive
            it->second.alpha_h[0][0] = this->prior[it->first] / it->second.forward_init(observation) ;
            it->second.results_h.likelihood = it->second.alpha_h[0][0] ;
            norm_const += it->second.alpha_h[0][0] ;
        }
        
        // Normalize Alpha variables
        for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
            int N = it->second.get_nbStates();
            for (int e=0 ; e<3 ; e++)
                for (int k=0 ; k<N ; k++)
                    it->second.alpha_h[e][k] /= norm_const;
        }
        
        model_iterator likeliestModel(this->models.begin());
        double maxLikelihood(0.0);
        int l(0);
        
        norm_const = 0.0;
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            if (it->second.results_h.likelihood > maxLikelihood) {
                likeliestModel = it;
                maxLikelihood = it->second.results_h.likelihood;
            }
            
            it->second.updateLikelihoodBuffer(it->second.results_h.likelihood);
            norm_const += it->second.results_h.likelihood;
            it->second.results_h.cumulativeLogLikelihood = it->second.cumulativeloglikelihood;
            modelLikelihoods[l++] = it->second.results_h.likelihood;
        }
        
        l = 0;
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.results_h.likelihoodnorm = it->second.results_h.likelihood / norm_const;
            modelLikelihoods[l++] = it->second.results_h.likelihoodnorm;
        }
        
        forwardInitialized = true;
        return likeliestModel;
    }
    
    model_iterator forward_update(float *observation, double *modelLikelihoods)
    {
        double norm_const(0.0) ;
        
        // TODO: is it useful to do it here? ==> better in play()
        // int dimension = this->get_dimension();
        
        // Frontier Algorithm: variables
        double tmp(0);
        vector<double> front; // frontier variable : intermediate computation variable
        
        // Intermediate variables: compute the sum of probabilities of making a transition to a new primitive
        likelihoodAlpha(1, V1);
        likelihoodAlpha(2, V2);
        
        // FORWARD UPDATE
        // --------------------------------------
        for (model_iterator dstit = this->models.begin(); dstit != this->models.end(); dstit++) {
            int N = dstit->second.get_nbStates();
            
            // 1) COMPUTE FRONTIER VARIABLE
            //    --------------------------------------
            front.resize(N) ;
            
            // k=0: first state of the primitive
            front[0] = dstit->second.transition[0] * dstit->second.alpha_h[0][0] ;
            
            int i(0);
            for (model_iterator srcit = this->models.begin(); srcit != this->models.end(); srcit++, i++) {
                front[0] += V1[i] * this->transition[srcit->first][dstit->first] + this->prior[dstit->first] * V2[i];
            }
            
            // k>0: rest of the primitive
            for (int k=1 ; k<N ; ++k)
            {
                front[k] = 0;
                for (int j = 0 ; j < N ; ++j)
                {
                    front[k] += dstit->second.transition[j*N+k] / (1 - dstit->second.exitProbabilities[j]) * dstit->second.alpha_h[0][j] ;
                }
            }
            
            for (int i=0; i<3; i++) {
                for (int k=0; k<N; k++){
                    dstit->second.alpha_h[i][k] = 0.0;
                }
            }
            
            // 2) UPDATE FORWARD VARIABLE
            //    --------------------------------------
            
            dstit->second.results_h.exitLikelihood = 0.0;
            dstit->second.results_h.likelihood = 0.0;
            
            // end of the primitive: handle exit states
            for (int k=0 ; k<N ; ++k)
            {
                tmp = dstit->second.obsProb(observation, k) * front[k];
                
                dstit->second.alpha_h[2][k] = this->exitTransition[dstit->first] * dstit->second.exitProbabilities[k] * tmp ;
                dstit->second.alpha_h[1][k] = (1 - this->exitTransition[dstit->first]) * dstit->second.exitProbabilities[k] * tmp ;
                dstit->second.alpha_h[0][k] = (1 - dstit->second.exitProbabilities[k]) * tmp;
                
                dstit->second.results_h.exitLikelihood += dstit->second.alpha_h[1][k];
                dstit->second.results_h.likelihood += dstit->second.alpha_h[0][k] + dstit->second.alpha_h[2][k] + dstit->second.results_h.exitLikelihood;
                
                norm_const += tmp;
            }
            
            // TODO: update cumulative + likelihood in circularBuffer
        }
        
        // Normalize Alpha variables
        for (model_iterator it = this->models.begin(); it != this->models.end(); it++) {
            int N = it->second.get_nbStates();
            for (int e=0 ; e<3 ; e++)
                for (int k=0 ; k<N ; k++)
                    it->second.alpha_h[e][k] /= norm_const;
        }
        
        model_iterator likeliestModel(this->models.begin());
        double maxLikelihood(0.0);
        int l(0.0);
        
        norm_const = 0.0;
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            if (it->second.results_h.likelihood > maxLikelihood) {
                likeliestModel = it;
                maxLikelihood = it->second.results_h.likelihood;
            }
            
            it->second.updateLikelihoodBuffer(it->second.results_h.likelihood);
            norm_const += it->second.results_h.likelihood;
            it->second.results_h.cumulativeLogLikelihood = it->second.cumulativeloglikelihood;
            modelLikelihoods[l++] = it->second.results_h.likelihood;
        }
        
        l = 0;
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.results_h.likelihoodnorm = it->second.results_h.likelihood / norm_const;
            modelLikelihoods[l++] = it->second.results_h.likelihoodnorm;
        }
        
        return likeliestModel;
    }
    
    /*!
     * @brief get instantaneous likelihood
     *
     * get instantaneous likelihood on alpha variable for exit state exitNum.
     * @param exitNum number of exit state (0=continue, 1=transition, 2=back to root). if -1, get likelihood over all exit states
     * @param likelihoodVector likelihood vector (size nbPrimitives)
     */
    void likelihoodAlpha(int exitNum, vector<double> &likelihoodVector) const
    {
        if (exitNum<0) { // Likelihood over all exit states
            int l(0);
            for (const_model_iterator it=this->models.begin(); it != this->models.end(); it++) {
                likelihoodVector[l] = 0.0;
                for (int exit = 0; exit<3; ++exit) {
                    for (int k=0; k<it->second.get_nbStates(); k++){
                        likelihoodVector[l] += it->second.alpha_h[exit][k];
                    }
                }
                l++;
            }
            
        } else { // Likelihood for exit state "exitNum"
            int l(0);
            for (const_model_iterator it=this->models.begin(); it != this->models.end(); it++) {
                likelihoodVector[l] = 0.0;
                for (int k=0; k<it->second.get_nbStates(); k++){
                    likelihoodVector[l] += it->second.alpha_h[exitNum][k];
                }
                l++;
            }
        }
    }
    
#pragma mark -
#pragma mark Playing
    virtual void initPlaying()
    {
        HierarchicalModel< HierarchicalHMMSubmodel<ownData>, Phrase<ownData, 1> >::initPlaying();
        
        int nbPrimitives = this->size();
        V1.resize(nbPrimitives, 0.0) ;
        V2.resize(nbPrimitives, 0.0) ;
        
        forwardInitialized = false;
    }
    
    virtual void play(float *obs, double *modelLikelihoods)
    {
        model_iterator likeliestModel;
        if (forwardInitialized) {
            likeliestModel = this->forward_update(obs, modelLikelihoods);
        } else {
            likeliestModel = this->forward_init(obs, modelLikelihoods);
        }
        
        // Compute time progression
        for (model_iterator it=this->models.begin(); it != this->models.end(); it++) {
            it->second.results_h.progress = centroid(it->second.alpha_h[0]);
        }
        
        // TODO: reintegrate Play mode?
        // TODO: reintegrate covariance determinants
    }
    
    HHMMResults getResults(Label classLabel)
    {
        if (this->models.find(classLabel) == this->models.end())
            throw RTMLException("Class Label Does not exist", __FILE__, __FUNCTION__, __LINE__);
        return this->models[classLabel].results_h;
    }
    
#pragma mark -
#pragma mark Protected attributes
protected:
    
    bool forwardInitialized;
    
    // Forward variables
    vector<double> V1 ;              //!< intermediate Forward variable
    vector<double> V2 ;              //!< intermediate Forward variable
    
#pragma mark -
#pragma mark Python
#ifdef SWIGPYTHON
public:
    void play(int dimension_, double *observation,
              int nbModels_, double *likelihoods,
              int nbModels__, double *cumulativelikelihoods)
    {
        int dimension = this->referenceModel.get_dimension();
        
        float *obs = new float[dimension];
        for (int d=0; d<dimension; d++) {
            obs[d] = float(observation[d]);
        }
        
        this->play(obs, likelihoods);
        
        int m(0);
        for (model_iterator it = this->models.begin(); it != this->models.end() ; it++)
            cumulativelikelihoods[m++] = it->second.cumulativeloglikelihood;
        
        delete[] obs;
    }
#endif
    
};


#endif
