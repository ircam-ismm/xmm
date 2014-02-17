//
//  hmm.h
//  mhmm
//
//  Created by Jules Francoise on 08/08/13.
//
//

#ifndef mhmm_hmm_h
#define mhmm_hmm_h

#include "gmm.h"

#define HMM_DEFAULT_NB_STATES 10
#define HMM_DEFAULT_ESTIMATEMEANS false

#define PLAY_EM_MAX_LOG_LIK_PERCENT_CHG 0.001
#define PLAY_EM_STEPS 5


/*!
 @enum TRANSITION_MODE
 Mode of transition of the model
 */
typedef enum _TRANSITION_MODE {
    ERGODIC,
    LEFT_RIGHT
} TRANSITION_MODE;

using namespace std;

/*!
 @struct HMMResults
 Structure containing the results of the recognition using the HMM
 */
struct HMMResults {
    double likelihood;                  //<! likelihood
    double cumulativeLogLikelihood;     //<! cumulative log-likelihood computed over a finite sliding window
    double progress;                    //<! Normalized time progression
};

template<bool ownData> class ConcurrentHMM;

/*!
 @class HMM
 @brief Hidden Markov Model
 @todo detailed description
 @tparam ownData defines if phrases has own data or shared memory
 */
template<bool ownData>
class HMM : public EMBasedLearningModel< Phrase<ownData, 1> >
{
    
    friend class ConcurrentHMM<ownData>;
    
public:
    typedef typename map<int, Phrase<ownData, 1>* >::iterator phrase_iterator;
    
    vector<double> alpha;
    HMMResults results;
    
#pragma mark -
#pragma mark Constructors
    /*! @name Constructors */
    /*!
     Constructor
     @param _trainingSet Training set associated with the model
     @param nbStates_ number of hidden states
     @param nbMixtureComponents_ number of gaussian mixture components for each state
     @param covarianceOffset_ offset added to the diagonal of covariances matrices to ensure convergence
     */
    HMM(TrainingSet< Phrase<ownData, 1> > *_trainingSet = NULL,
        int nbStates_ = HMM_DEFAULT_NB_STATES,
        int nbMixtureComponents_ = GMM_DEFAULT_NB_MIXTURE_COMPONENTS,
        float covarianceOffset_ = GMM_DEFAULT_COVARIANCE_OFFSET)
    : EMBasedLearningModel< Phrase<ownData, 1> >(_trainingSet)
    {
        nbStates = nbStates_;
        
        if (this->trainingSet) {
            dimension = this->trainingSet->get_dimension();
        } else {
            dimension = 1;
        }
        
        nbMixtureComponents = nbMixtureComponents_;
        covarianceOffset = covarianceOffset_;
        
        reallocParameters();
        
        for (int i=0; i<nbStates; i++) {
            states[i].set_trainingSet(_trainingSet);
        }
        
        play_EM_stopCriterion.minSteps = PLAY_EM_STEPS;
        play_EM_stopCriterion.maxSteps = 0;
        play_EM_stopCriterion.percentChg = PLAY_EM_MAX_LOG_LIK_PERCENT_CHG;
        
        transitionMode = LEFT_RIGHT;
        
        estimateMeans = HMM_DEFAULT_ESTIMATEMEANS;
        
        initTraining();
    }
    
    /*!
     Copy constructor
     */
    HMM(HMM const& src) : EMBasedLearningModel< Phrase<ownData, 1> >(src)
    {
        _copy(this, src);
    }
    
    /*!
     Assignment
     */
    HMM& operator=(HMM const& src)
    {
        if(this != &src)
        {
            _copy(this, src);
        }
        return *this;
    }
    
    /*!
     Copy between 2 MHMM models (called by copy constructor and assignment methods)
     */
    using EMBasedLearningModel< Phrase<ownData, 1> >::_copy;
    virtual void _copy(HMM *dst, HMM const& src)
    {
        EMBasedLearningModel< Phrase<ownData, 1> >::_copy(dst, src);
        dst->nbMixtureComponents     = src.nbMixtureComponents;
        dst->covarianceOffset        = src.covarianceOffset;
        dst->nbStates = src.nbStates;
        dst->estimateMeans = src.estimateMeans;
        
        dst->alpha.resize(dst->nbStates);
        dst->previousAlpha.resize(dst->nbStates);
        dst->beta.resize(dst->nbStates);
        dst->previousBeta.resize(dst->nbStates);
        
        dst->dimension = src.dimension;
        
        dst->transition = src.transition;
        dst->prior = src.prior;
        dst->transitionMode = src.transitionMode;
        
        dst->states = src.states;
        dst->play_EM_stopCriterion = src.play_EM_stopCriterion;
    }
    
    /*!
     Destructor
     */
    virtual ~HMM()
    {
        prior.clear();
        transition.clear();
        alpha.clear();
        previousAlpha.clear();
        beta.clear();
        previousBeta.clear();
        states.clear();
    }
    
#pragma mark -
#pragma mark Connection to Training set
    /*! @name Connection to Training set */
    /*!
     handle notifications of the training set
     
     here only the dimensions attributes of the training set are considered
     */
    void notify(string attribute)
    {
        if (!this->trainingSet) return;
        
        for (int i=0; i<nbStates; i++) {
            states[i].notify(attribute);
        }
        
        if (attribute == "dimension") {
            dimension = this->trainingSet->get_dimension();
            return;
        }
    }
    
    /*!
     Set training set associated with the model
     */
    void set_trainingSet(TrainingSet< Phrase<ownData, 1> > *_trainingSet)
    {
        this->trainingSet = _trainingSet;
        if (this->trainingSet) {
            dimension = this->trainingSet->get_dimension();
        }
        
        for (int i=0; i<nbStates; i++) {
            states[i].set_trainingSet(_trainingSet);
        }
    }
    
#pragma mark -
#pragma mark Parameters initialization
    /*! @name Parameters initialization */
    /*!
     Re-allocate model parameters
     */
    void reallocParameters()
    {
        prior.resize(nbStates);
        transition.resize(nbStates*nbStates);
        alpha.resize(nbStates);
        previousAlpha.resize(nbStates);
        beta.resize(nbStates);
        previousBeta.resize(nbStates);
        states.resize(nbStates, GMM<ownData>(this->trainingSet, nbMixtureComponents, covarianceOffset));
    }
    
    /*!
     evaluate the number of hidden states based on the length of the training examples
     @todo integrate state factor as attribute
     */
    void evaluateNbStates()
    {
        int factor = 5;
        if (!this->trainingSet) return;
        int nbPhrases = this->trainingSet->size();
        if (nbPhrases == 0) return;
        this->set_nbStates(((*this->trainingSet)(0))->second->length() / factor);
    }
    
    /*!
     initialize model parameters to their default values
     */
    virtual void initParametersToDefault()
    {
        for (int i=0; i<nbStates; i++) {
            states[i].initParametersToDefault();
        }
    }
    
    /*!
     initialize the means of each state with the first phrase (single gaussian)
     */
    void initMeansWithFirstPhrase()
    {
        if (!this->trainingSet) return;
        int nbPhrases = this->trainingSet->size();
        if (nbPhrases == 0) return;
        
        for (int n=0; n<nbStates; n++)
            for (int d=0; d<dimension; d++)
                states[n].mean[d] = 0.0;
        
        vector<int> factor(nbStates, 0);
        int step = ((*this->trainingSet)(0))->second->length() / nbStates;
        int offset(0);
        for (int n=0; n<nbStates; n++) {
            for (int t=0; t<step; t++) {
                for (int d=0; d<dimension; d++) {
                    states[n].mean[d] += (*((*this->trainingSet)(0)->second))(offset+t, d);
                }
            }
            offset += step;
            factor[n] += step;
        }
        
        for (int n=0; n<nbStates; n++)
            for (int d=0; d<dimension; d++)
                states[n].mean[d] /= factor[n];
    }
    
    /*!
     initialize the means of each state with all training phrases (single gaussian)
     */
    void initMeansWithAllPhrases_single()
    {
        if (!this->trainingSet) return;
        int nbPhrases = this->trainingSet->size();
        if (nbPhrases == 0) return;
        
        for (int n=0; n<nbStates; n++)
            for (int d=0; d<dimension; d++)
                states[n].mean[d] = 0.0;
        
        vector<int> factor(nbStates, 0);
        for (int i=0; i<nbPhrases; i++) {
            int step = ((*this->trainingSet)(i))->second->length() / nbStates;
            int offset(0);
            for (int n=0; n<nbStates; n++) {
                for (int t=0; t<step; t++) {
                    for (int d=0; d<dimension; d++) {
                        states[n].mean[d] += (*((*this->trainingSet)(i)->second))(offset+t, d);
                    }
                }
                offset += step;
                factor[n] += step;
            }
        }
        
        for (int n=0; n<nbStates; n++)
            for (int d=0; d<dimension; d++)
                states[n].mean[d] /= factor[n];
    }
    
    /*!
     initialize the covariances of each state with all training phrases (single gaussian)
     */
    void initCovariancesWithAllPhrases_single()
    {
        if (!this->trainingSet) return;
        int nbPhrases = this->trainingSet->size();
        if (nbPhrases == 0) return;
        
        for (int n=0; n<nbStates; n++)
            for (int d1=0; d1<dimension; d1++)
                for (int d2=0; d2<dimension; d2++)
                    states[n].covariance[d1*dimension+d2] = -states[n].mean[d1]*states[n].mean[d2];
        
        vector<int> factor(nbStates, 0);
        for (int i=0; i<nbPhrases; i++) {
            int step = ((*this->trainingSet)(i))->second->length() / nbStates;
            int offset(0);
            for (int n=0; n<nbStates; n++) {
                for (int t=0; t<step; t++) {
                    for (int d1=0; d1<dimension; d1++) {
                        for (int d2=0; d2<dimension; d2++) {
                            states[n].covariance[d1*dimension+d2] += (*((*this->trainingSet)(i)->second))(offset+t, d1) * (*((*this->trainingSet)(i)->second))(offset+t, d2);
                        }
                    }
                }
                offset += step;
                factor[n] += step;
            }
        }
        
        for (int n=0; n<nbStates; n++)
            for (int d1=0; d1<dimension; d1++)
                for (int d2=0; d2<dimension; d2++)
                    states[n].covariance[d1*dimension+d2] /= factor[n];
    }
    
    /*!
     initialize the means of each states with all training phrases (mixture of gaussian)
     */
    void initMeansWithAllPhrases_mixture()
    {
        if (!this->trainingSet) return;
        int nbPhrases = this->trainingSet->size();
        if (nbPhrases == 0) return;
        
        for (int i=0; i<min(nbPhrases, nbMixtureComponents); i++) {
            int step = ((*this->trainingSet)(i))->second->length() / nbStates;
            int offset(0);
            for (int n=0; n<nbStates; n++) {
                for (int d=0; d<dimension; d++) {
                    states[n].meanOfComponent(i)[d] = 0.0;
                }
                for (int t=0; t<step; t++) {
                    for (int d=0; d<dimension; d++) {
                        states[n].meanOfComponent(i)[d] += (*((*this->trainingSet)(i)->second))(offset+t, d) / float(step);
                    }
                }
                offset += step;
            }
        }
    }
    
    /*!
     initialize the covariances of each states with all training phrases (mixture of gaussian)
     */
    void initCovariancesWithAllPhrases_mixture()
    {
        if (!this->trainingSet) return;
        int nbPhrases = this->trainingSet->size();
        if (nbPhrases == 0) return;
        
        for (int i=0; i<min(nbPhrases, nbMixtureComponents); i++) {
            int step = ((*this->trainingSet)(i))->second->length() / nbStates;
            int offset(0);
            for (int n=0; n<nbStates; n++) {
                for (int d1=0; d1<dimension; d1++) {
                    for (int d2=0; d2<dimension; d2++) {
                        states[n].covarianceOfComponent(i)[d1*dimension+d2] = -states[n].meanOfComponent(i)[d1]*states[n].meanOfComponent(i)[d2];
                    }
                }
                for (int t=0; t<step; t++) {
                    for (int d1=0; d1<dimension; d1++) {
                        for (int d2=0; d2<dimension; d2++) {
                            states[n].covarianceOfComponent(i)[d1*dimension+d2] += (*((*this->trainingSet)(i)->second))(offset+t, d1) * (*((*this->trainingSet)(i)->second))(offset+t, d2) / float(step);
                        }
                    }
                }
                offset += step;
            }
        }
    }
    
    /*!
     set the prior and transition matrix to ergodic
     */
    void setErgodic()
    {
        for (int i=0 ; i<nbStates; i++) {
            prior[i] = 1/(float)nbStates;
            for (int j=0; j<nbStates; j++) {
                transition[i*nbStates+j] = 1/(float)nbStates;
            }
        }
    }
    
    /*!
     set the prior and transition matrix to left-right (no state skip)
     */
    void setLeftRight()
    {
        for (int i=0 ; i<nbStates; i++) {
            prior[i] = 0.;
            for (int j=0; j<nbStates; j++) {
                transition[i*nbStates+j] = ((i == j) || ((i+1) == j)) ? 0.5 : 0;
            }
        }
        transition[nbStates*nbStates-1] = 1.;
        prior[0] = 1.;
    }
    
    /*!
     Normalize transition probabilities
     */
    void normalizeTransitions()
    {
        double norm_prior(0.), norm_transition;
        for (int i=0; i<nbStates; i++) {
            norm_prior += prior[i];
            norm_transition = 0.;
            for (int j=0; j<nbStates; j++)
                norm_transition += transition[i*nbStates+j];
            for (int j=0; j<nbStates; j++)
                transition[i*nbStates+j] /= norm_transition;
        }
        for (int i=0; i<nbStates; i++)
            prior[i] /= norm_prior;
    }
    
#pragma mark -
#pragma mark Accessors
    /*! @name Accessors */
    int get_dimension() const
    {
        return dimension;
    }
    
    virtual int get_nbStates() const
    {
        return nbStates;
    }
    
    void set_nbStates(int nbStates_)
    {
        if (nbStates_ < 1) throw RTMLException("Number of states must be > 0", __FILE__, __FUNCTION__, __LINE__);;
        if (nbStates_ == nbStates) return;
        
        nbStates = nbStates_;
        reallocParameters();
        
        this->trained = false;
    }
    
    int get_nbMixtureComponents() const
    {
        return nbMixtureComponents;
    }
    
    void set_nbMixtureComponents(int nbMixtureComponents_)
    {
        if (nbMixtureComponents_ == nbMixtureComponents) return;
        
        for (int i=0; i<nbStates; i++) {
            states[i].set_nbMixtureComponents(nbMixtureComponents_);
        }
        
        nbMixtureComponents = nbMixtureComponents_;
        
        this->trained = false;
    }
    
    float  get_covarianceOffset() const
    {
        return covarianceOffset;
    }
    
    void set_covarianceOffset(float covarianceOffset_)
    {
        if (covarianceOffset_ == covarianceOffset) return;
        
        for (int i=0; i<nbStates; i++) {
            states[i].set_covarianceOffset(covarianceOffset_);
        }
        covarianceOffset = covarianceOffset_;
    }
    
    /*!
     get transition mode of the hidden Markov Chain
     @return string correpsonding to the transition mode (left-right / ergodic)
     @todo: remove transitionMode to simplify forward complexity (cf gf)
     */
    string get_transitionMode() const
    {
        if (transitionMode == ERGODIC) {
            return "ergodic";
        } else {
            return "left-right";
        }
    }
    
    /*!
     set transition mode of the hidden Markov Chain
     @param transMode_str string keyword correpsonding to the transition mode (left-right / ergodic)
     */
    void set_transitionMode(string transMode_str)
    {
        if (!transMode_str.compare("ergodic")) {
            transitionMode = ERGODIC;
        } else if (!transMode_str.compare("left-right")) {
            transitionMode = LEFT_RIGHT;
        } else {
            throw RTMLException("Wrong Transition mode. choose 'ergodic' or 'left-right'", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    
#pragma mark -
#pragma mark Observation probabilities
    /*! @name Observation probabilities */
    /*!
     Gaussian observation probability
     @param obs observation vector
     @param stateNumber index of the state
     @param mixtureComponent index of the gaussian mixture component (full mixture observation probability if unspecified)
     */
    double obsProb(const float *obs, int stateNumber, int mixtureComponent=-1)
    {
        return states[stateNumber].obsProb(obs, mixtureComponent);
    }
    
#pragma mark -
#pragma mark Forward-Backward algorithm
    /*! @name Forward-Backward Algorithm */
    /*!
     initialization of the forward algorithm
     @param obs observation vector
     @return likelihood
     */
    double forward_init(const float *obs)
    {
        double norm_const(0.);
        for (int i=0 ; i<nbStates ; i++) {
            alpha[i] = prior[i] * obsProb(obs, i);
            norm_const += alpha[i];
        }
        if (norm_const > 0) {
            for (int i=0 ; i<nbStates ; i++) {
                alpha[i] /= norm_const;
            }
            return 1/norm_const;
        } else {
            for (int j=0; j<nbStates; j++) {
                alpha[j] = 1./double(nbStates);
            }
            return 1.;
        }
    }
    
    /*!
     update of the forward algorithm
     @param obs observation vector
     @return likelihood
     */
    double forward_update(const float *obs)
    {
        double norm_const(0.);
        previousAlpha = alpha;
        for (int j=0; j<nbStates; j++) {
            alpha[j] = 0.;
            for (int i=0; i<nbStates; i++) {
                alpha[j] += previousAlpha[i] * transition[i*nbStates+j];
            }
            alpha[j] *= obsProb(obs, j);
            norm_const += alpha[j];
        }
        if (norm_const > 0) {
            for (int j=0; j<nbStates; j++) {
                alpha[j] /= norm_const;
            }
            return 1./norm_const;
        } else {
            for (int j=0; j<nbStates; j++) {
                alpha[j] = 1./double(nbStates);
            }
            return 1.;
        }
    }
    
    /*!
     backward initialization
     @param ct inverse of the likelihood at time step t computed with the forward algorithm (see Rabiner 1989)
     */
    void backward_init(double ct)
    {
        for (int i=0 ; i<nbStates ; i++)
            beta[i] = ct;
    }
    
    /*!
     backward update
     @param obs observation vector at time t
     @param ct inverse of the likelihood at time step t computed with the forward algorithm (see Rabiner 1989)
     */
    void backward_update(const float *obs, double ct)
    {
        previousBeta = beta;
        for (int i=0 ; i<nbStates; i++) {
            beta[i] = 0.;
            for (int j=0; j<nbStates; j++) {
                beta[i] += transition[i*nbStates+j]
                * previousBeta[j]
                * obsProb(obs, j);
            }
            beta[i] *= ct;
            if (isnan(beta[i]) || isinf(abs(beta[i]))) {
                beta[i] = 1e100;
            }
        }
    }
    
#pragma mark -
#pragma mark Training algorithm
    /*! @name Training Algorithm */
    /*!
     Initialization of the parameters before training
     */
    void initTraining()
    {
        // Initialize Model Parameters
        // ---------------------------------------
        if (transitionMode == ERGODIC) {
            setErgodic();
        } else {
            setLeftRight();
        }
        for (int i=0; i<nbStates; i++) {
            states[i].initTraining();
        }
        
        if (!this->trainingSet) return;
        
        if (nbMixtureComponents > 1) {
            initMeansWithAllPhrases_mixture();
            initCovariancesWithAllPhrases_mixture();
        } else {
            // initMeansWithAllPhrases_single();
            initMeansWithFirstPhrase();
            initCovariancesWithAllPhrases_single();
        }
        this->trained = false;
        
        int nbPhrases = this->trainingSet->size();
        
        
        // Initialize Algorithm variables
        // ---------------------------------------
        gammaSequence.resize(nbPhrases);
        epsilonSequence.resize(nbPhrases);
        gammaSequencePerMixture.resize(nbPhrases);
        int maxT(0);
        int i(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
            int T = it->second->length();
            gammaSequence[i].resize(T*nbStates);
            epsilonSequence[i].resize(T*nbStates*nbStates);
            gammaSequencePerMixture[i].resize(nbMixtureComponents);
            for (int c=0; c<nbMixtureComponents; c++) {
                gammaSequencePerMixture[i][c].resize(T*nbStates);
            }
            if (T>maxT) {
                maxT = T;
            }
            i++;
        }
        alpha_seq.resize(maxT*nbStates);
        beta_seq.resize(maxT*nbStates);
        
        gammaSum.resize(nbStates);
        gammaSumPerMixture.resize(nbStates*nbMixtureComponents);
    }
    
    /*!
     termination of the training algorithm
     */
    void finishTraining()
    {
        normalizeTransitions();
        LearningModel< Phrase<ownData, 1> >::finishTraining();
    }
    
    /*!
     update method of the EM algorithm (calls baumWelch_update)
     */
    virtual double train_EM_update()
    {
        return baumWelch_update();
    }
    
    /*!
     Baum-Welch update for Hidden Markov Models
     */
    double baumWelch_update()
    {
        double log_prob(0.);
        
        // Forward-backward for each phrase
        // ********************************************
        int phraseIndex(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
            log_prob += baumWelch_forwardBackward(it->second, phraseIndex++);
        }
        
        baumWelch_gammaSum();
        
        // Re-estimate model parameters
        // ********************************************
        
        // set covariance and mixture coefficients to zero for each state
        for (int i=0; i<nbStates; i++) {
            states[i].setParametersToZero();
        }
        
        baumWelch_estimateMixtureCoefficients();
        if (estimateMeans)
            baumWelch_estimateMeans();
        
        baumWelch_estimateCovariances();
        if (transitionMode == ERGODIC)
            baumWelch_estimatePrior();
        baumWelch_estimateTransitions();
        
        return log_prob;
    }
    
    double baumWelch_forwardBackward(Phrase<ownData, 1>* currentPhrase, int phraseIndex)
    {
        int T = currentPhrase->length();
        
        vector<double> ct(T);
        
        vector<double>::iterator alpha_seq_it = alpha_seq.begin();
        
        double log_prob;
        
        // Forward algorithm
        ct[0] = forward_init(currentPhrase->get_dataPointer(0));
        log_prob = -log(ct[0]);
        copy(alpha.begin(), alpha.end(), alpha_seq_it);
        // vectorCopy(alpha_seq_it, alpha.begin(), nbStates);
        alpha_seq_it += nbStates;
        
        for (int t=1; t<T; t++) {
            ct[t] = forward_update(currentPhrase->get_dataPointer(t));
            log_prob -= log(ct[t]);
            copy(alpha.begin(), alpha.end(), alpha_seq_it);
            // vectorCopy(alpha_seq_it, alpha.begin(), nbStates);
            alpha_seq_it += nbStates;
        }
        
        // Backward algorithm
        backward_init(ct[T-1]);
        vector<double>::iterator beta_seq_it = beta_seq.begin()+(T-1)*nbStates;
        copy(beta.begin(), beta.end(), beta_seq_it);
        // vectorCopy(beta_seq_it, beta.begin(), nbStates);
        beta_seq_it -= nbStates;
        
        for (int t=T-2; t>=0; t--) {
            backward_update(currentPhrase->get_dataPointer(t+1),
                            ct[t]);
            copy(beta.begin(), beta.end(), beta_seq_it);
            // vectorCopy(beta_seq_it, beta.begin(), nbStates);
            beta_seq_it -= nbStates;
        }
        
        // Compute Gamma Variable
        for (int t=0; t<T; t++) {
            for (int i=0; i<nbStates; i++) {
                gammaSequence[phraseIndex][t*nbStates+i] = alpha_seq[t*nbStates+i] * beta_seq[t*nbStates+i] / ct[t];
            }
        }
        
        // Compute Gamma variable for each mixture component
        double oo;
        double norm_const;
        
        for (int t=0; t<T; t++) {
            for (int i=0; i<nbStates; i++) {
                norm_const = 0.;
                for (int c=0; c<nbMixtureComponents; c++) {
                    oo = obsProb(currentPhrase->get_dataPointer(t),
                                 i,
                                 c);
                    gammaSequencePerMixture[phraseIndex][c][t*nbStates+i] = gammaSequence[phraseIndex][t*nbStates+i] * oo;
                    norm_const += oo;
                }
                if (norm_const > 0)
                    for (int c=0; c<nbMixtureComponents; c++)
                        gammaSequencePerMixture[phraseIndex][c][t*nbStates+i] /= norm_const;
            }
        }
        
        // Compute Epsilon Variable
        for (int t=0; t<T-1; t++) {
            for (int i=0; i<nbStates; i++) {
                for (int j=0; j<nbStates; j++) {
                    epsilonSequence[phraseIndex][t*nbStates*nbStates+i*nbStates+j] = alpha_seq[t*nbStates+i]
                    * transition[i*nbStates+j]
                    * obsProb(currentPhrase->get_dataPointer(t+1),
                              j)
                    * beta_seq[(t+1)*nbStates+j];
                }
            }
        }
        
        return log_prob;
    }
    
    void baumWelch_gammaSum()
    {
        for (int i=0; i<nbStates; i++) {
            gammaSum[i] = 0.;
            for (int c=0; c<nbMixtureComponents; c++) {
                gammaSumPerMixture[i*nbMixtureComponents+c] = 0.;
            }
        }
        
        int phraseLength;
        int phraseIndex(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
            phraseLength = it->second->length();
            for (int i=0; i<nbStates; i++) {
                for (int t=0; t<phraseLength; t++) {
                    gammaSum[i] += gammaSequence[phraseIndex][t*nbStates+i];
                    for (int c=0; c<nbMixtureComponents; c++) {
                        gammaSumPerMixture[i*nbMixtureComponents+c] += gammaSequencePerMixture[phraseIndex][c][t*nbStates+i];
                    }
                }
            }
            phraseIndex++;
        }
    }
    
    void baumWelch_estimateMixtureCoefficients()
    {
        int phraseLength;
        int phraseIndex(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
        {
            phraseLength = it->second->length();
            for (int i=0; i<nbStates; i++) {
                for (int t=0; t<phraseLength; t++) {
                    for (int c=0; c<nbMixtureComponents; c++) {
                        states[i].mixtureCoeffs[c] += gammaSequencePerMixture[phraseIndex][c][t*nbStates+i];
                    }
                }
            }
            phraseIndex++;
        }
        
        // Scale mixture coefficients
        for (int i=0; i<nbStates; i++) {
            states[i].normalizeMixtureCoeffs();
        }
    }
    
    void baumWelch_estimateMeans()
    {
        int phraseLength;
        // Re-estimate Mean
        int phraseIndex(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
        {
            phraseLength = it->second->length();
            for (int i=0; i<nbStates; i++) {
                for (int c=0; c<nbMixtureComponents; c++) {
                    for (int d=0; d<dimension; d++) {
                        states[i].meanOfComponent(c)[d] = 0.0;
                    }
                }
                for (int t=0; t<phraseLength; t++) {
                    for (int c=0; c<nbMixtureComponents; c++) {
                        for (int d=0; d<dimension; d++) {
                            states[i].meanOfComponent(c)[d] += gammaSequencePerMixture[phraseIndex][c][t*nbStates+i] * (*it->second)(t, d);
                        }
                    }
                }
            }
            phraseIndex++;
        }
        
        // Normalize mean
        for (int i=0; i<nbStates; i++) {
            for (int c=0; c<nbMixtureComponents; c++) {
                if (gammaSumPerMixture[i*nbMixtureComponents+c] > 0) {
                    for (int d=0; d<dimension; d++) {
                        states[i].meanOfComponent(c)[d] /= gammaSumPerMixture[i*nbMixtureComponents+c];
                    }
                }
            }
        }
    }
    
    void baumWelch_estimateCovariances()
    {
        int phraseLength;
        
        int phraseIndex(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
        {
            phraseLength = it->second->length();
            for (int i=0; i<nbStates; i++) {
                for (int t=0; t<phraseLength; t++) {
                    for (int c=0; c<nbMixtureComponents; c++) {
                        for (int d1=0; d1<dimension; d1++) {
                            for (int d2=0; d2<dimension; d2++) {
                                states[i].covarianceOfComponent(c)[d1*dimension+d2] += gammaSequencePerMixture[phraseIndex][c][t*nbStates+i]
                                * ((*it->second)(t, d1) - states[i].meanOfComponent(c)[d1])
                                * ((*it->second)(t, d2) - states[i].meanOfComponent(c)[d2]);
                            }
                        }
                    }
                }
            }
            phraseIndex++;
        }
        
        // Scale covariance
        for (int i=0; i<nbStates; i++) {
            for (int c=0; c<nbMixtureComponents; c++) {
                if (gammaSumPerMixture[i*nbMixtureComponents+c] > 0) {
                    for (int d=0; d<dimension*dimension; d++) {
                        states[i].covarianceOfComponent(c)[d] /= gammaSumPerMixture[i*nbMixtureComponents+c];
                    }
                }
            }
            states[i].addCovarianceOffset();
            states[i].updateInverseCovariances();
        }
    }
    
    void baumWelch_estimatePrior()
    {
        // Set prior vector to 0
        for (int i=0; i<nbStates; i++)
            prior[i] = 0.;
        
        // Re-estimate Prior probabilities
        double sumprior = 0.;
        int phraseIndex(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
        {
            for (int i=0; i<nbStates; i++) {
                prior[i] += gammaSequence[phraseIndex][i];
                sumprior += gammaSequence[phraseIndex][i];
            }
            phraseIndex++;
        }
        
        // Scale Prior vector
        if (sumprior == 0) {
            cout << "sumprior == 0" << endl;
        }
        for (int i=0; i<nbStates; i++) {
            prior[i] /= sumprior;
        }
    }
    
    void baumWelch_estimateTransitions()
    {
        // Set prior vector and transition matrix to 0
        for (int i=0; i<nbStates; i++)
            for (int j=0; j<nbStates; j++)
                transition[i*nbStates+j] = 0.;
        
        int phraseLength;
        // Re-estimate Prior and Transition probabilities
        int phraseIndex(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
        {
            phraseLength = it->second->length();
            for (int i=0; i<nbStates; i++) {
                for (int j=0; j<nbStates; j++)
                {
                    for (int t=0; t<phraseLength-1; t++) {
                        transition[i*nbStates+j] += epsilonSequence[phraseIndex][t*nbStates*nbStates+i*nbStates+j];
                    }
                }
            }
            phraseIndex++;
        }
        
        // Scale transition matrix
        for (int i=0; i<nbStates; i++) {
            if (gammaSum[i] > 0)
                for (int j=0; j<nbStates; j++)
                    transition[i*nbStates+j] /= gammaSum[i];
        }
    }
    
#pragma mark -
#pragma mark Play!
    /*! @name Playing */
    void initPlaying()
    {
        EMBasedLearningModel< Phrase<ownData, 1> >::initPlaying();
        forwardInitialized = false;
    }
    
    void addCyclicTransition(double proba)
    {
        transition[(nbStates-1)*nbStates] = proba; // Add Cyclic Transition probability
    }
    
    /*!
     @brief play function: estimate sound observation given current gesture frame
     @param obs pointer to current observation vector \n!!! Must be of size dimension (gesture + sound features)
     @return likelihood computed on the gesture modality by a forwad algorithm
     */
    double play(float *obs)
    {
        double ct;
        
        if (forwardInitialized) {
            ct = forward_update(obs);
        } else {
            this->likelihoodBuffer.clear();
            ct = forward_init(obs);
        }
        
        forwardInitialized = true;
        
        results.likelihood = 1./ct;
        this->updateLikelihoodBuffer(results.likelihood);
        results.cumulativeLogLikelihood = this->cumulativeloglikelihood;
        results.progress = centroid(alpha);
        
        return results.likelihood;
    }
    
    
    HMMResults getResults()
    {
        return results;
    }
        
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    /*!
     Write to JSON Node
     */
    virtual JSONNode to_json() const
    {
        JSONNode json_hmm(JSON_NODE);
        json_hmm.set_name("HMM");
        
        // Write Parent: EM Learning Model
        JSONNode json_emmodel = EMBasedLearningModel< Phrase<ownData, 1> >::to_json();
        json_emmodel.set_name("parent");
        json_hmm.push_back(json_emmodel);
        
        // Scalar Attributes
        json_hmm.push_back(JSONNode("dimension", dimension));
        json_hmm.push_back(JSONNode("nbStates", nbStates));
        json_hmm.push_back(JSONNode("nbMixtureComponents", nbMixtureComponents));
        json_hmm.push_back(JSONNode("covarianceOffset", covarianceOffset));
        json_hmm.push_back(JSONNode("transitionMode", int(transitionMode)));
        
        // Model Parameters
        json_hmm.push_back(vector2json(prior, "prior"));
        json_hmm.push_back(vector2json(transition, "transition"));
        
        // States
        JSONNode json_states(JSON_ARRAY);
        for (int i=0 ; i<nbStates ; i++)
        {
            json_states.push_back(states[i].to_json());
        }
        json_states.set_name("states");
        json_hmm.push_back(json_states);
        
        return json_hmm;
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
            EMBasedLearningModel< Phrase<ownData, 1> >::from_json(*root_it);
            root_it++;
            
            // Get Dimension
            assert(root_it != root.end());
            assert(root_it->name() == "dimension");
            assert(root_it->type() == JSON_NUMBER);
            dimension = root_it->as_int();
            root_it++;
            
            // Get Number of states
            assert(root_it != root.end());
            assert(root_it->name() == "nbStates");
            assert(root_it->type() == JSON_NUMBER);
            nbStates = root_it->as_int();
            root_it++;
            
            // Get Number of Mixture Components
            assert(root_it != root.end());
            assert(root_it->name() == "nbMixtureComponents");
            assert(root_it->type() == JSON_NUMBER);
            nbMixtureComponents = root_it->as_int();
            root_it++;
            
            // Get Covariance Offset
            assert(root_it != root.end());
            assert(root_it->name() == "covarianceOffset");
            assert(root_it->type() == JSON_NUMBER);
            covarianceOffset = root_it->as_float();
            root_it++;
            
            // Get Transition Mode
            assert(root_it != root.end());
            assert(root_it->name() == "transitionMode");
            assert(root_it->type() == JSON_NUMBER);
            transitionMode = TRANSITION_MODE(root_it->as_int());
            root_it++;
            
            // Reallocate model parameters
            reallocParameters();
            
            // Get Prior
            assert(root_it != root.end());
            assert(root_it->name() == "prior");
            assert(root_it->type() == JSON_ARRAY);
            json2vector(*root_it, prior, nbStates);
            root_it++;
            
            // Get Mean
            assert(root_it != root.end());
            assert(root_it->name() == "transition");
            assert(root_it->type() == JSON_ARRAY);
            json2vector(*root_it, transition, nbStates*nbStates);
            root_it++;
            
            // Get States
            assert(root_it != root.end());
            assert(root_it->name() == "states");
            assert(root_it->type() == JSON_ARRAY);
            for (int i=0 ; i<nbStates ; i++) {
                states[i].from_json((*root_it)[i]);
            }
            
        } catch (exception &e) {
            throw RTMLException("Error reading JSON, Node: " + root.name() + " >> " + e.what());
        }
        
        this->trained = true;
    }
    
    /*!
     Write model to stream
     @todo check if all attributes are written
     @param outStream output stream
     */
    void write(ostream& outStream)
    {
        // TODO: check if all attributes are written
        outStream << "# HMM \n";
        outStream << "# =========================================\n";
        EMBasedLearningModel< Phrase<ownData, 1> >::write(outStream);
        outStream << "# Dimension\n";
        outStream << dimension << endl;
        outStream << "# Number of states\n";
        outStream << nbStates << endl;
        outStream << "# Transition Mode\n";
        outStream << transitionMode << endl;
        outStream << "# Number of mixture Components\n";
        outStream << nbMixtureComponents << endl;
        outStream << "# Covariance Offset\n";
        outStream << covarianceOffset << endl;
        outStream << "# estimate means in EM\n";
        outStream << estimateMeans << endl;
        outStream << "# Prior probabilities\n";
        for (int i=0 ; i<nbStates; i++) {
            outStream << prior[i] << " ";
        }
        outStream << endl;
        outStream << "# Transition probabilities\n";
        for (int i=0 ; i<nbStates; i++) {
            for (int j=0 ; j<nbStates; j++) {
                outStream << transition[i*nbStates+j] << " ";
            }
            outStream << endl;
        }
        outStream << "# States\n";
        for (int i=0; i<nbStates; i++) {
            states[i].write(outStream);
        }
    }
    
    /*!
     Read model from stream
     @todo check if all attributes are written
     @param inStream input stream
     */
    void read(istream& inStream)
    {
        EMBasedLearningModel< Phrase<ownData, 1> >::read(inStream);
        
        // Get Dimensions
        skipComments(&inStream);
        inStream >> dimension;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        // Get Number of states
        skipComments(&inStream);
        int nbStates_;
        inStream >> nbStates_;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        set_nbStates(nbStates_);
        
        // Get transition Mode
        skipComments(&inStream);
        int _tm;
        inStream >> _tm;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        transitionMode = TRANSITION_MODE(_tm);
        
        // Get Number of mixture components
        skipComments(&inStream);
        int nbMixtureComponents_;
        inStream >> nbMixtureComponents_;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        set_nbMixtureComponents(nbMixtureComponents_);
        
        // Get Covariance Offset
        skipComments(&inStream);
        inStream >> covarianceOffset;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        // Get Estimate Means
        skipComments(&inStream);
        inStream >> estimateMeans;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        // Get Prior probabilities
        skipComments(&inStream);
        for (int i=0 ; i<nbStates; i++) {
            inStream >> prior[i];
            if (!inStream.good())
                throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        }
        
        // Get transition probabilities
        skipComments(&inStream);
        for (int i=0 ; i<nbStates; i++) {
            for (int j=0 ; j<nbStates; j++) {
                inStream >> transition[i*nbStates+j];
                if (!inStream.good())
                    throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
            }
        }
        
        // Get States
        for (int i=0 ; i<nbStates; i++) {
            states[i].read(inStream);
        }
        
        // cout << "file read and model trained\n" << endl;
        this->trained = true;
    }
    
#pragma mark -
#pragma mark Python
    /*! @name Python methods */
#ifdef SWIGPYTHON
    double play(int dimension_, double *observation,
                int nbStates_, double *alpha_)
    {
        float *obs = new float[dimension];
        for (int d=0; d<dimension; d++) {
            obs[d] = float(observation[d]);
        }
        
        double likelihood = play(obs);
        
        delete[] obs;
        
        for (int i=0; i<nbStates_; i++) {
            alpha_[i] = alpha[i];
        }
        
        return likelihood;
    }
#endif
    
#pragma mark -
#pragma mark Debug
    /*! @name Debug */
    void dump()
    {
        if (this->trainingSet) {
            int nbPhrases = this->trainingSet->size();
            cout << "Number of phrases = " << nbPhrases << endl;
            for (phrase_iterator it = this->trainingSet->begin() ; it != this->trainingSet->end() ; it++) {
                cout << "size of phrase " << it->first << " = " << it->second->length() << endl;
                // cout << "phrase " << it->first << ": data = \n";
                // it->second->print();
            }
            cout << "\n\n";
        }
        
        cout << "number of states = " << nbStates << endl;
        cout << "Dimension = " << dimension << endl;
        cout << "number of mixture components = " << nbMixtureComponents << endl;
        cout << "covariance offset = " << covarianceOffset << endl;
        cout << "prior probabilities : ";
        for (int i=0; i<nbStates; i++)
            cout << prior[i] << " ";
        cout << "\n\n";
        cout << "transition probabilities :\n";
        for (int i=0; i<nbStates; i++) {
            for (int j=0; j<nbStates; j++)
                cout << transition[i*nbStates+j] << " ";
            cout << endl;
        }
        cout << "\n\n";
        cout << "mixture mixtureCoeffs:\n";
        for (int i=0; i<nbStates; i++) {
            for (int c=0; c<nbMixtureComponents; c++)
                cout << states[i].mixtureCoeffs[c] << " ";
            cout << endl;
        }
        cout << "\n";
        cout << "mean (state per state):\n";
        vector<float>::iterator mean_it;
        for (int i=0; i<nbStates; i++) {
            mean_it = states[i].mean.begin();
            for (int c=0; c<nbMixtureComponents; c++) {
                for (int d=0; d<dimension; d++) {
                    cout << *(mean_it++) << " ";
                }
                cout << "\n";
            }
            cout << "\n";
        }
        cout << "\n";
        cout << "covariance (state per state):\n";
        vector<float>::iterator cov_it;
        for (int i=0; i<nbStates; i++) {
            cov_it = states[i].covariance.begin();
            for (int c=0; c<nbMixtureComponents; c++) {
                for (int d1=0; d1<dimension; d1++) {
                    for (int d2=0; d2<dimension; d2++) {
                        cout << *(cov_it++) << " ";
                    }
                    cout << "\n";
                }
                cout << "\n";
            }
            cout << "\n";
        }
        cout << "\n";
    }
    
#pragma mark -
#pragma mark Protected Attributes
    /*! @name Protected Attributes */
protected:
    // Model parameters
    // ================================
    int nbStates;
    int dimension;
    
    vector<float> transition;
    vector<float> prior;
    TRANSITION_MODE transitionMode;
    
    vector<GMM<ownData> > states;
    int    nbMixtureComponents;
    float  covarianceOffset;       //<! offset on covariance [= prior] (added to obs prob re-estimation)
    
    vector<double> previousAlpha;
    vector<double> beta;
    vector<double> previousBeta;
    
    EMStopCriterion play_EM_stopCriterion;
    
    // EM algorithm variables
    // ================================
    bool forwardInitialized;
    bool estimateMeans;
    vector< vector<double> > gammaSequence;
    vector< vector<double> > epsilonSequence;
    vector< vector< vector<double> > > gammaSequencePerMixture;
    vector<double> alpha_seq;
    vector<double> beta_seq;
    vector<double> gammaSum;
    vector<double> gammaSumPerMixture;
};


#endif
