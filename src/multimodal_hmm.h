//
//  mhmm_lib.h
//  mhmm
//
//  Created by Jules Francoise on 16/10/12.
//
//

/*!
 * @mainpage
 *
 * author  : Jules Francoise\n
 * Contact : jules.francoise@ircam.fr
 *
 * @section description
 * TODO
 *
 @endcode
 *
 */


#ifndef __mhmm__mhmm_lib__
#define __mhmm__mhmm_lib__

#include "multimodal_gmm.h"
#include "hmm.h"

#define MHMM_DEFAULT_NB_STATES 10
#define MHMM_DEFAULT_ESTIMATEMEANS false

#define PLAY_EM_MAX_LOG_LIK_PERCENT_CHG 0.001
#define PLAY_EM_STEPS 5

using namespace std;

/*!
 @struct MHMMResults
 Structure containing the results of the prediction using the multimodal HMM
 */
struct MHMMResults {
    vector<float> observation_sound;    //<! estimated sound observation
    vector<float> covariance_sound;     //<! covariance estimated on the sound
    double covarianceDeterminant_sound; //<! determinant of the sound covariance
    double likelihood;                  //<! likelihood
    double cumulativeLogLikelihood;     //<! cumulative log-likelihood computed over a finite sliding window
};

template<bool ownData> class ConcurrentMHMM;

/*!
 @class MultimodalHMM
 @brief Multimodal Hidden Markov Model
 @todo detailed description
 @tparam ownData defines if phrases has own data or shared memory
 */
template<bool ownData>
class MultimodalHMM : public EMBasedLearningModel<GestureSoundPhrase<ownData>, int>
{
    
    friend class ConcurrentMHMM<ownData>;
    
public:
    typedef typename map<int, GestureSoundPhrase<ownData>* >::iterator phrase_iterator;
    
    vector<double> alpha;
    MHMMResults results;
    
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
    MultimodalHMM(TrainingSet<GestureSoundPhrase<ownData>, int> *_trainingSet=NULL,
                  int nbStates_ = MHMM_DEFAULT_NB_STATES,
                  int nbMixtureComponents_ = MGMM_DEFAULT_NB_MIXTURE_COMPONENTS,
                  float covarianceOffset_ = MGMM_DEFAULT_COVARIANCE_OFFSET)
    : EMBasedLearningModel<GestureSoundPhrase<ownData>, int>(_trainingSet)
    {
        nbStates = nbStates_;
        
        transition.resize(nbStates*nbStates);
        prior.resize(nbStates);
        
        alpha.resize(nbStates);
        previousAlpha.resize(nbStates);
        beta.resize(nbStates);
        previousBeta.resize(nbStates);
        
        if (this->trainingSet) {
            dimension_gesture = this->trainingSet->get_dimension_gesture();
            dimension_sound = this->trainingSet->get_dimension_sound();
        } else {
            dimension_gesture = 0;
            dimension_sound = 0;
        }
        dimension_total = dimension_gesture + dimension_sound;
        
        nbMixtureComponents = nbMixtureComponents_;
        covarianceOffset = covarianceOffset_;
        
        states.resize(nbStates, MultimodalGMM<ownData>(NULL, nbMixtureComponents, covarianceOffset));
        
        for (int i=0; i<nbStates; i++) {
            states[i].set_trainingSet(_trainingSet);
        }
        
        play_EM_stopCriterion.minSteps = PLAY_EM_STEPS;
        play_EM_stopCriterion.maxSteps = 0;
        play_EM_stopCriterion.percentChg = PLAY_EM_MAX_LOG_LIK_PERCENT_CHG;
        
        transitionMode = LEFT_RIGHT;
        
        estimateMeans = MHMM_DEFAULT_ESTIMATEMEANS;
        
        initTraining();
    }
    
    /*!
     Copy constructor
     */
    MultimodalHMM(MultimodalHMM const& src) : EMBasedLearningModel< GestureSoundPhrase<ownData>, int>(src)
    {
        _copy(this, src);
    }
    
    /*!
     Assignment
     */
    MultimodalHMM& operator=(MultimodalHMM const& src)
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
    using EMBasedLearningModel<GestureSoundPhrase<ownData>, int>::_copy;
    virtual void _copy(MultimodalHMM *dst, MultimodalHMM const& src)
    {
        EMBasedLearningModel<GestureSoundPhrase<ownData>, int>::_copy(dst, src);
        dst->nbMixtureComponents     = src.nbMixtureComponents;
        dst->covarianceOffset        = src.covarianceOffset;
        dst->nbStates = src.nbStates;
        dst->estimateMeans = src.estimateMeans;
        
        dst->alpha.resize(dst->nbStates);
        dst->previousAlpha.resize(dst->nbStates);
        dst->beta.resize(dst->nbStates);
        dst->previousBeta.resize(dst->nbStates);
        
        dst->dimension_gesture = src.dimension_gesture;
        dst->dimension_sound = src.dimension_sound;
        dst->dimension_total = dst->dimension_gesture + dst->dimension_sound;
        
        dst->transition = src.transition;
        dst->prior = src.prior;
        dst->transitionMode = src.transitionMode;
        
        dst->states = src.states;
        dst->play_EM_stopCriterion = src.play_EM_stopCriterion;
    }
    
    /*!
     Destructor
     */
    virtual ~MultimodalHMM()
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
        
        if (attribute == "dimension_gesture") {
            dimension_gesture = this->trainingSet->get_dimension_gesture();
            dimension_total = dimension_gesture + dimension_sound;
            // reallocParameters();
            return;
        }
        if (attribute == "dimension_sound") {
            dimension_sound = this->trainingSet->get_dimension_sound();
            dimension_total = dimension_gesture + dimension_sound;
            // reallocParameters();
            return;
        }
    }
    
    /*!
     Set training set associated with the model
     */
    void set_trainingSet(TrainingSet<GestureSoundPhrase<ownData>, int> *_trainingSet)
    {
        this->trainingSet = _trainingSet;
        if (this->trainingSet) {
            dimension_gesture = this->trainingSet->get_dimension_gesture();
            dimension_sound = this->trainingSet->get_dimension_sound();
        }
        dimension_total = dimension_gesture + dimension_sound;
        
        for (int i=0; i<nbStates; i++) {
            states[i].set_trainingSet(_trainingSet);
        }
        // TODO: Maybe the training set is not handle properly for each state.
        // Need to segment phrases...
    }
    
#pragma mark -
#pragma mark Parameters initialization
    /*! @name Parameters initialization */
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
        this->set_nbStates(((*this->trainingSet)(0))->second->getlength() / factor);
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
     initialize the means of each state with all training phrases (single gaussian)
     */
    void initMeansWithAllPhrases_single()
    {
        if (!this->trainingSet) return;
        int nbPhrases = this->trainingSet->size();
        if (nbPhrases == 0) return;
        
        for (int n=0; n<nbStates; n++)
            for (int d=0; d<dimension_total; d++)
                states[n].mean[d] = 0.0;
        
        vector<int> factor(nbStates, 0);
        for (int i=0; i<nbPhrases; i++) {
            int step = ((*this->trainingSet)(i))->second->getlength() / nbStates;
            int offset(0);
            for (int n=0; n<nbStates; n++) {
                for (int t=0; t<step; t++) {
                    for (int d=0; d<dimension_total; d++) {
                        states[n].mean[d] += (*((*this->trainingSet)(i)->second))(offset+t, d);
                    }
                }
                offset += step;
                factor[n] += step;
            }
        }
        
        for (int n=0; n<nbStates; n++)
            for (int d=0; d<dimension_total; d++)
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
            for (int d1=0; d1<dimension_total; d1++)
                for (int d2=0; d2<dimension_total; d2++)
                    states[n].covariance[d1*dimension_total+d2] = -states[n].mean[d1]*states[n].mean[d2];
        
        vector<int> factor(nbStates, 0);
        for (int i=0; i<nbPhrases; i++) {
            int step = ((*this->trainingSet)(i))->second->getlength() / nbStates;
            int offset(0);
            for (int n=0; n<nbStates; n++) {
                for (int t=0; t<step; t++) {
                    for (int d1=0; d1<dimension_total; d1++) {
                        for (int d2=0; d2<dimension_total; d2++) {
                            states[n].covariance[d1*dimension_total+d2] += (*((*this->trainingSet)(i)->second))(offset+t, d1) * (*((*this->trainingSet)(i)->second))(offset+t, d2);
                        }
                    }
                }
                offset += step;
                factor[n] += step;
            }
        }
        
        for (int n=0; n<nbStates; n++)
            for (int d1=0; d1<dimension_total; d1++)
                for (int d2=0; d2<dimension_total; d2++)
                    states[n].covariance[d1*dimension_total+d2] /= factor[n];
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
            int step = ((*this->trainingSet)(i))->second->getlength() / nbStates;
            int offset(0);
            for (int n=0; n<nbStates; n++) {
                for (int d=0; d<dimension_total; d++) {
                    states[n].meanOfComponent(i)[d] = 0.0;
                }
                for (int t=0; t<step; t++) {
                    for (int d=0; d<dimension_total; d++) {
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
            int step = ((*this->trainingSet)(i))->second->getlength() / nbStates;
            int offset(0);
            for (int n=0; n<nbStates; n++) {
                for (int d1=0; d1<dimension_total; d1++) {
                    for (int d2=0; d2<dimension_total; d2++) {
                        states[n].covarianceOfComponent(i)[d1*dimension_total+d2] = -states[n].meanOfComponent(i)[d1]*states[n].meanOfComponent(i)[d2];
                    }
                }
                for (int t=0; t<step; t++) {
                    for (int d1=0; d1<dimension_total; d1++) {
                        for (int d2=0; d2<dimension_total; d2++) {
                            states[n].covarianceOfComponent(i)[d1*dimension_total+d2] += (*((*this->trainingSet)(i)->second))(offset+t, d1) * (*((*this->trainingSet)(i)->second))(offset+t, d2) / float(step);
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
    int get_dimension_gesture() const
    {
        return dimension_gesture;
    }
    
    int get_dimension_sound() const
    {
        return dimension_sound;
    }
    
    virtual int get_nbStates() const
    {
        return nbStates;
    }
    
    void set_nbStates(int nbStates_)
    {
        if (nbStates_ < 1) throw RTMLException("Number of states must be > 0", __FILE__, __FUNCTION__, __LINE__);;
        if (nbStates_ == nbStates) return;
        
        prior.resize(nbStates_);
        transition.resize(nbStates_*nbStates_);
        states.resize(nbStates_, MultimodalGMM<ownData>(this->trainingSet, nbMixtureComponents, covarianceOffset));
        alpha.resize(nbStates_);
        previousAlpha.resize(nbStates_);
        beta.resize(nbStates_);
        previousBeta.resize(nbStates_);
        
        nbStates = nbStates_;
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
        this->trained = false;
        nbMixtureComponents = nbMixtureComponents_;
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
#pragma mark DEPRECATED: EM for playing
    /*! @name DEPRECATED: EM for playing */
    int get_play_EM_minSteps() const
    {
        return play_EM_stopCriterion.minSteps;
    }
    
    int get_play_EM_maxSteps() const
    {
        return play_EM_stopCriterion.maxSteps;
    }
    
    double get_play_EM_percentChange() const
    {
        return play_EM_stopCriterion.percentChg;
    }
    
    void set_play_EM_minSteps(int minsteps)
    {
        if (minsteps < 1) throw RTMLException("Minimum number of EM steps must be > 0", __FILE__, __FUNCTION__, __LINE__);
        play_EM_stopCriterion.minSteps = minsteps;
    }
    
    void set_play_EM_maxSteps(int maxsteps)
    {
        if (maxsteps < 1) throw RTMLException("Maximum number of EM steps must be > 0", __FILE__, __FUNCTION__, __LINE__);
        play_EM_stopCriterion.maxSteps = maxsteps;
    }
    
    void set_play_EM_maxLogLikPercentChg(double logLikPercentChg_)
    {
        if (logLikPercentChg_ > 0) {
            play_EM_stopCriterion.percentChg = logLikPercentChg_;
        } else {
            throw RTMLException("Max loglikelihood difference for EM stop criterion must be > 0", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    
#pragma mark -
#pragma mark Observation probabilities
    /*! @name Observation probabilities */
    /*!
     Multimodal gaussian observation probability
     @param obs_gesture gesture observation vector
     @param obs_sound sound observation vector
     @param stateNumber index of the state
     @param mixtureComponent index of the gaussian mixture component (full mixture observation probability if unspecified)
     */
    double obsProb_gestureSound(const float *obs_gesture, const float *obs_sound, int stateNumber, int mixtureComponent=-1)
    {
        return states[stateNumber].obsProb_gestureSound(obs_gesture, obs_sound, mixtureComponent);
    }
    
    /*!
     Unimodal gaussian observation probability (gesture only)
     @param obs_gesture gesture observation vector
     @param stateNumber index of the state
     */
    double obsprob_gesture(const float *obs_gesture, int stateNumber)
    {
        return states[stateNumber].obsProb_gesture(obs_gesture);
    }
    
    /*!
     Unimodal gaussian observation probability (sound only)
     @param obs_sound sound observation vector
     @param stateNumber index of the state
     */
    double obsProb_sound(const float *obs_sound, int stateNumber)
    {
        return states[stateNumber].obsProb_sound(obs_sound);
    }
    
#pragma mark -
#pragma mark Forward-Backward algorithm
    /*! @name Forward-Backward Algorithm */
    /*!
     initialization of the forward algorithm
     if gestureAndSound is true, the forward algorithm is computed with multimodal data
     @param obs_gesture gesture observation vector
     @param obs_sound sound observation vector (optional)
     @return likelihood
     */
    double forward_init(const float *obs_gesture, const float *obs_sound=NULL)
    {
        double norm_const(0.);
        for (int i=0 ; i<nbStates ; i++) {
            alpha[i] = prior[i] * ((gestureAndSound && obs_sound) ? obsProb_gestureSound(obs_gesture, obs_sound, i) : obsprob_gesture(obs_gesture, i));
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
     if gestureAndSound is true, the forward algorithm is computed with multimodal data
     @param obs_gesture gesture observation vector
     @param obs_sound sound observation vector (optional)
     @return likelihood
     */
    double forward_update(const float *obs_gesture, const float *obs_sound=NULL)
    {
        double norm_const(0.);
        if (!keepPreviousAlpha)
            previousAlpha = alpha;
        for (int j=0; j<nbStates; j++) {
            alpha[j] = 0.;
            for (int i=0; i<nbStates; i++) {
                alpha[j] += previousAlpha[i] * transition[i*nbStates+j];
            }
            if (gestureAndSound && obs_sound) {
                alpha[j] *= obsProb_gestureSound(obs_gesture, obs_sound, j);
            } else {
                alpha[j] *= obsprob_gesture(obs_gesture, j);
            }
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
     foward update with estimated sound observation 
     @warning generally unused in current version of max/python implementations
     */
    double forward_update_withNewObservation(const float *obs_gesture, const float *obs_sound)
    {
        keepPreviousAlpha = true;
        gestureAndSound = true;
        if (forwardInitialized)
            return forward_update(obs_gesture, obs_sound);
        else
            return forward_init(obs_gesture, obs_sound);
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
     backward update on multimodal data
     @param obs_gesture gesture observation vector at time t
     @param obs_sound sound observation vector at time t
     @param ct inverse of the likelihood at time step t computed with the forward algorithm (see Rabiner 1989)
     */
    void backward_update(const float *obs_gesture, const float *obs_sound, double ct)
    {
        previousBeta = beta;
        for (int i=0 ; i<nbStates; i++) {
            beta[i] = 0.;
            for (int j=0; j<nbStates; j++) {
                beta[i] += transition[i*nbStates+j]
                * previousBeta[j]
                * ((gestureAndSound && obs_sound) ? obsProb_gestureSound(obs_gesture, obs_sound, j) : obsprob_gesture(obs_gesture, j));
            }
            beta[i] *= ct;
            if (isnan(beta[i]) || isinf(abs(beta[i]))) {
                beta[i] = 1e100;
                // cout << "beta["<<i<<"] = " << beta[i] << endl;
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
            initMeansWithAllPhrases_single();
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
            int T = it->second->getlength();
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
    
    double baumWelch_forwardBackward(GestureSoundPhrase<ownData>* currentPhrase, int phraseIndex)
    {
        int T = currentPhrase->getlength();
        
        vector<double> ct(T);
        
        vector<double>::iterator alpha_seq_it = alpha_seq.begin();
        
        double log_prob;
        
        gestureAndSound = true;
        keepPreviousAlpha = false;
        
        // Forward algorithm
        ct[0] = forward_init(currentPhrase->get_dataPointer_gesture(0),
                             currentPhrase->get_dataPointer_sound(0));
        log_prob = -log(ct[0]);
        copy(alpha.begin(), alpha.end(), alpha_seq_it);
        // vectorCopy(alpha_seq_it, alpha.begin(), nbStates);
        alpha_seq_it += nbStates;
        
        for (int t=1; t<T; t++) {
            ct[t] = forward_update(currentPhrase->get_dataPointer_gesture(t),
                                   currentPhrase->get_dataPointer_sound(t));
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
            backward_update(currentPhrase->get_dataPointer_gesture(t+1),
                            currentPhrase->get_dataPointer_sound(t+1),
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
                    oo = obsProb_gestureSound(currentPhrase->get_dataPointer_gesture(t),
                                              currentPhrase->get_dataPointer_sound(t),
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
                    * obsProb_gestureSound(currentPhrase->get_dataPointer_gesture(t+1),
                                           currentPhrase->get_dataPointer_sound(t+1),
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
            phraseLength = it->second->getlength();
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
            phraseLength = it->second->getlength();
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
            phraseLength = it->second->getlength();
            for (int i=0; i<nbStates; i++) {
                for (int c=0; c<nbMixtureComponents; c++) {
                    for (int d=0; d<dimension_total; d++) {
                        states[i].meanOfComponent(c)[d] = 0.0;
                    }
                }
                for (int t=0; t<phraseLength; t++) {
                    for (int c=0; c<nbMixtureComponents; c++) {
                        for (int d=0; d<dimension_total; d++) {
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
                    for (int d=0; d<dimension_total; d++) {
                        states[i].meanOfComponent(c)[d] /= gammaSumPerMixture[i*nbMixtureComponents+c];
                    }
                }
            }
        }
    }
    
    void baumWelch_estimateCovariances()
    {
        int phraseLength;
        int dimension_total = dimension_gesture + dimension_sound;
        
        int phraseIndex(0);
        for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
        {
            phraseLength = it->second->getlength();
            for (int i=0; i<nbStates; i++) {
                for (int t=0; t<phraseLength; t++) {
                    for (int c=0; c<nbMixtureComponents; c++) {
                        for (int d1=0; d1<dimension_total; d1++) {
                            for (int d2=0; d2<dimension_total; d2++) {
                                states[i].covarianceOfComponent(c)[d1*dimension_total+d2] += gammaSequencePerMixture[phraseIndex][c][t*nbStates+i]
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
                    for (int d=0; d<dimension_total*dimension_total; d++) {
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
            phraseLength = it->second->getlength();
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
        EMBasedLearningModel<GestureSoundPhrase<ownData>, int>::initPlaying();
        forwardInitialized = false;
        for (int i=0; i<nbStates; i++) {
            states[i].estimateConditionalSoundCovariance();
        }
        
        results.covariance_sound.resize(dimension_sound*dimension_sound);
        results.observation_sound.resize(dimension_sound);
    }
    
    void addCyclicTransition(double proba)
    {
        transition[(nbStates-1)*nbStates] = proba; // Add Cyclic Transition probability
    }
    
    /*!
     @brief play function: estimate sound observation given current gesture frame
     @param obs pointer to current observation vector \n!!! Must be of size dimension_total (gesture + sound features)
     @return likelihood computed on the gesture modality by a forwad algorithm
     */
    double play(float *obs)
    {
        double ct;
        double obs_prob(-log(0.)), old_obs_prob;
        keepPreviousAlpha = false;
        gestureAndSound = false;
        int n(0);
        
        do
        {
            old_obs_prob = obs_prob;
            if (forwardInitialized) {
                ct = forward_update(obs, obs+dimension_gesture);
            } else {
                this->likelihoodBuffer.clear();
                ct = forward_init(obs, obs+dimension_gesture);
            }
            obs_prob = log(ct);
            
            //cout << "step "<< n << ": precent-change = " << 100.*fabs((obs_prob-old_obs_prob)/old_obs_prob) << "logProb = " << obs_prob << endl;
            
            estimateObservation(obs);
            //estimateObservationByLikeliestState(obs);
            
            n++;
            keepPreviousAlpha = true;
            gestureAndSound = true;
        } while (false);//(!play_EM_stop(n, obs_prob, old_obs_prob));
        // TODO: test iterative estimation on live gestures
        
        forwardInitialized = true;
        
        for (int i=0; i<dimension_sound; i++) {
            results.observation_sound[i] = obs[dimension_gesture+i];
        }
        
        estimateCovariance();
        // TODO: put estimateCovariance as an attribute of the object
        
        results.likelihood = 1./ct;
        this->updateLikelihoodBuffer(results.likelihood);
        results.cumulativeLogLikelihood = this->cumulativeloglikelihood;
        
        return results.likelihood;
    }
    
    /*!
     estimate sound observation bvector as a weighted sum of gaussian mixture regressions for each state
     @param obs pointer to current observation vector \n!!! Must be of size dimension_total (gesture + sound features)
     */
    virtual void estimateObservation(float *obs)
    {
        float *obs2 = new float[dimension_total];
        copy(obs, obs+dimension_total, obs2);
        for (int d=0; d<dimension_sound; d++) {
            obs[dimension_gesture+d] = 0.0;
        }
        for (int i=0; i<nbStates; i++) {
            states[i].regression(obs2);
            for (int d=0; d<dimension_sound; d++) {
                obs[dimension_gesture+d] += alpha[i] * obs2[dimension_gesture+d];
            }
        }
    }
    
    MHMMResults getResults()
    {
        return results;
    }
    
#pragma mark -
#pragma mark Experimental
    /*! @name Experimental */
    void estimateCovariance()
    {
        for (int d1=0; d1<dimension_sound; d1++) {
            for (int d2=0; d2<dimension_sound; d2++) {
                results.covariance_sound[d1*dimension_sound+d2] = 0.0;
                for (int i=0; i<nbStates; i++) {
                    results.covariance_sound[d1*dimension_sound+d2] += alpha[i] * alpha[i] * states[i].covariance_sound[d1*dimension_sound + d2];
                }
            }
        }
        
        // Compute determinant
        Matrix<float> cov_matrix(dimension_sound, dimension_sound, false);
        Matrix<float> *inverseMat;
        double det;
        cov_matrix.data = results.covariance_sound.begin();
        inverseMat = cov_matrix.pinv(&det);
        results.covarianceDeterminant_sound = det;
        delete inverseMat;
    }
    
    void estimateObservationByLikeliestState(float *obs)
    {
        states[likeliestState()].regression(obs);
    }
    
    int likeliestState()
    {
        int state(0);
        double stateLikelihood(alpha[0]);
        for (int i=0; i<nbStates; i++) {
            if (alpha[i] > stateLikelihood) {
                state = i;
                stateLikelihood = alpha[i];
            }
        }
        return state;
    }
    
    bool play_EM_stop(int step, double obs_prob, double old_obs_prob)
    {
        if (play_EM_stopCriterion.maxSteps < play_EM_stopCriterion.minSteps)
        {
            return (step >= play_EM_stopCriterion.maxSteps);
        }
        else
        {
            return (step >= play_EM_stopCriterion.minSteps) || (fabs((obs_prob - old_obs_prob) / obs_prob) < play_EM_stopCriterion.percentChg);
        }
    }
    
#pragma mark -
#pragma mark File IO
    /*! @name File IO */
    /*!
     Write model to stream
     @todo check if all attributes are written
     @param outStream output stream
     @param writeTrainingSet defines if the training set needs to be written with the model
     */
    void write(ostream& outStream, bool writeTrainingSet=false)
    {
        // TODO: check if all attributes are written
        outStream << "# Multimodal HMM \n";
        outStream << "# =========================================\n";
        EMBasedLearningModel<GestureSoundPhrase<ownData>, int>::write(outStream, writeTrainingSet);
        outStream << "# Dimensions\n";
        outStream << dimension_gesture << " " << dimension_sound << endl;
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
     @param readTrainingSet defines if the training set needs to be read with the model
     */
    void read(istream& inStream, bool readTrainingSet=false)
    {
        EMBasedLearningModel<GestureSoundPhrase<ownData>, int>::read(inStream, readTrainingSet);
        
        // Get Dimensions
        skipComments(&inStream);
        inStream >> dimension_gesture;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        inStream >> dimension_sound;
        if (!inStream.good())
            throw RTMLException("Error reading file: wrong format", __FILE__, __FUNCTION__, __LINE__);
        
        dimension_total = dimension_gesture + dimension_sound;
        
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
    double play(int dimension_gesture_, double *observation_gesture,
                int dimension_sound_, double *observation_sound_out,
                int nbStates_, double *alpha_,
                int dimension_sound_square, double *outCovariance)
    {
        float *observation_total = new float[dimension_total];
        for (int d=0; d<dimension_gesture; d++) {
            observation_total[d] = float(observation_gesture[d]);
        }
        for (int d=0; d<dimension_sound; d++)
            observation_total[d+dimension_gesture] = 0.;
        
        double likelihood = play(observation_total);
        
        for (int d=0; d<dimension_sound_; d++) {
            observation_sound_out[d] = double(observation_total[dimension_gesture+d]);
        }
        delete[] observation_total;
        
        for (int i=0; i<nbStates_; i++) {
            alpha_[i] = alpha[i];
        }
        
        for (int d=0; d<dimension_sound_square; d++) {
            outCovariance[d] = results.covariance_sound[d];
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
                cout << "size of phrase " << it->first << " = " << it->second->getlength() << endl;
                // cout << "phrase " << it->first << ": data = \n";
                // it->second->print();
            }
            cout << "\n\n";
        }
        
        cout << "number of states = " << nbStates << endl;
        cout << "Dimensions = " << dimension_gesture << " " << dimension_sound << endl;
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
                for (int d=0; d<dimension_total; d++) {
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
                for (int d1=0; d1<dimension_total; d1++) {
                    for (int d2=0; d2<dimension_total; d2++) {
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
    int dimension_gesture;
    int dimension_sound;
    int dimension_total;
    
    vector<float> transition;
    vector<float> prior;
    TRANSITION_MODE transitionMode;
    
    vector<MultimodalGMM<ownData> > states;
    int    nbMixtureComponents;
    bool   gestureAndSound;        //<! defines if obsprob must be computed with sound (else: only gesture)
    float  covarianceOffset;       //<! offset on covariance [= prior] (added to obs prob re-estimation)
    
    vector<double> previousAlpha;
    vector<double> beta;
    vector<double> previousBeta;
    
    bool keepPreviousAlpha; // If true, the alpha variable in memory is not copied in previousAlpha (useful for EM during PLaying)
    
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