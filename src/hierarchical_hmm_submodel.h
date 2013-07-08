//
//  hierarchical_hmm_submodel.h
//  mhmm
//
//  Created by Jules Francoise on 02/07/13.
//
//

#ifndef mhmm_hierarchical_hmm_submodel_h
#define mhmm_hierarchical_hmm_submodel_h

#include "em_based_learning_model.h"

#define HHMM_DEFAULT_SIGMA 0.1
#define HHMM_DEFAULT_EXITPROBABILITY 0.1

using namespace std;
using namespace rtml;

struct HHMMResults {
    int     internalState;          //!< internal state (high level)
	int     productionState;        //!< production state (low level --> sample level)
	int     exitState;              //!< exit state (0 = continue; 1 = high level transition; 2 = end of gesture)
	double  timenorm;               //!< Time progression (relative time progression in segment)
	double  maxlikelihood;          //!< likelihood of the likeliest state
	vector<double> likelihood;      //!< likelihood vector (instantaneaous)
	vector<double> cumLikelihood;   //!< likelihood vector (cumulative)
	vector<double> likelihoodNorm;  //!< likelihood vector (instantaneaous + normalized)
	vector<double> exitlikelihood;
	HHMMResults()
	{
		internalState   = 0;
		exitState       = 0;
		productionState = 0;
		timenorm        = 0;
		maxlikelihood   = 0;
		likelihood      = 0;
		cumLikelihood   = 0;
		likelihoodNorm  = 0;
		exitlikelihood  = 0;
	}
	~HHMMResults()
	{
		likelihood.clear();
		cumLikelihood.clear();
		likelihoodNorm.clear();
        exitlikelihood.clear();
	}
};

/*!
 * @class HierarchicalHMMSubmodel
 * @brief Template HMM model => submodel of Hierarchical HMM
 */
template<bool ownData>
class HierarchicalHMMSubmodel : public EMBasedLearningModel<GesturePhrase<ownData>, int>
{
public:
    typedef typename map<int, GesturePhrase<ownData>* >::iterator phrase_iterator;
    
    vector<double> alpha;
    HHMMResults results;
    
#pragma mark -
#pragma mark Constructors
    HierarchicalHMMSubmodel(TrainingSet<GesturePhrase<ownData>, int> *_trainingSet=NULL,
                            float sigma_ = HHMM_DEFAULT_SIGMA)
    : EMBasedLearningModel<GesturePhrase<ownData>, int>(_trainingSet)
    {
        if (this->trainingSet) {
            dimension_gesture = this->trainingSet->get_dimension_gesture();
            dimension_sound = this->trainingSet->get_dimension_sound();
        } else {
            dimension_gesture = 0;
            dimension_sound = 0;
        }
        dimension_total = dimension_gesture + dimension_sound;
        
        sigma = sigma_;
        initTraining();
    }
    
    HierarchicalHMMSubmodel(HierarchicalHMMSubmodel const& src) : EMBasedLearningModel< GesturePhrase<ownData>, int>(src)
    {
        copy(this, src);
    }
    
    HierarchicalHMMSubmodel& operator=(HierarchicalHMMSubmodel const& src)
    {
        if(this != &src)
        {
            copy(this, src);
        }
        return *this;
    }
    
    virtual void copy(HierarchicalHMMSubmodel *dst, HierarchicalHMMSubmodel const& src)
    {
        EMBasedLearningModel<GesturePhrase<ownData>, int>::copy(dst, src);
        
        dst->dimension_gesture = src.dimension_gesture;
        dst->dimension_sound = src.dimension_sound;
        dst->dimension_total = dst->dimension_gesture + dst->dimension_sound;
        
        dst->sigma = src.sigma;
        dst->exitProbabilities = src.exitProbabilities;
        dst->LR_transition = src.LR_transition;
    }
    
    ~MultimodalHMM()
    {
        exitProbabilities.clear();
        LR_transition.clear();
        alpha.clear();
    }
    
#pragma mark -
#pragma mark Connection to Training set
    void notify(string attribute)
    {
        if (!this->trainingSet) return;
        
        if (attribute == "dimension_gesture") {
            dimension_gesture = this->trainingSet->get_dimension_gesture();
            dimension_total = dimension_gesture + dimension_sound;
            return;
        }
        if (attribute == "dimension_sound") {
            dimension_sound = this->trainingSet->get_dimension_sound();
            dimension_total = dimension_gesture + dimension_sound;
            return;
        }
    }
    
    void set_trainingSet(TrainingSet<GesturePhrase<ownData>, int> *_trainingSet)
    {
        this->trainingSet = _trainingSet;
        if (this->trainingSet) {
            dimension_gesture = this->trainingSet->get_dimension_gesture();
            dimension_sound = this->trainingSet->get_dimension_sound();
        }
        dimension_total = dimension_gesture + dimension_sound;
    }
    
#pragma mark -
#pragma mark Accessors
    int get_dimension_gesture() const
    {
        return dimension_gesture;
    }
    
    int get_dimension_sound() const
    {
        return dimension_sound;
    }
    
    float  get_sigma() const
    {
        return sigma;
    }
    
    void set_sigma(float sigma_)
    {
        sigma = sigma_;
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
#pragma mark Training algorithm
    void initTraining()
    {
        LR_transition.resize(4);
        LR_transition[0] = 0.2
        LR_transition[1] = 0.5
        LR_transition[2] = 0.2
        LR_transition[3] = 0.1
        updateExitProbabilities();
    }
    
    void finishTraining()
    {}
    
    virtual int train()
    {
        return 0;
    }
    
    virtual double train_EM_update()
    {
        return 0.;
    }
    
#pragma mark -
#pragma mark Play!
    double obsprob_gesture(const float *obs_gesture, int stateNumber)
    {
        double a(0);
        phrase_iterator p = this->trainingSet.begin();
        
        for (int d = 0; d<dimension; ++d)
        {
            a -= pow((obs_gesture[d] - double((*(p->second))(stateNumber, d)) , 2);
        }
        
        a /= (2 * (pow(sigma, 2.))) ;
        return exp(a) ;
    }
    
    void initPlaying()
    {
        if (this->trainingSet && this->trainingSet.size() > 0)
        {
            int primitiveSize = this->trainingSet.begin()->second->getlength();
            alpha.resize(primitiveSize);
        }
        else
        {
            throw RTMLException("Training set is empty", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    
    /*!
     * @brief play function: estimate sound observation given current gesture frame
     *
     * The observation is estimated iteratively (Expectation-Maximization)
     *
     * @param obs pointer to current observation vector \n!!! Must be of size dimension_total (gesture + sound features)
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
        
        results.likelihood = this->updateLikelihoodBuffer(1./ct);
        results.cumulativeLogLikelihood = this->cumulativeloglikelihood;
        
        return results.likelihood;
    }
    
    virtual void estimateObservation(float *obs)
    {
        float *obs2 = new float[dimension_total];
        memcpy(obs2, obs, dimension_total*sizeof(float));
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
    
    /*
     EXPERIMENTAL
     */
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
        if (play_EM_stopCriterion.type == STEPS)
        {
            return (step >= play_EM_stopCriterion.steps);
        }
        else if (play_EM_stopCriterion.type == PERCENT_CHG)
        {
            return (fabs((obs_prob - old_obs_prob) / obs_prob) < play_EM_stopCriterion.percentChg);
        }
        else // play_EM_stopCriterion == BOTH
        {
            return (step >= play_EM_stopCriterion.steps) || (fabs((obs_prob - old_obs_prob) / obs_prob) < play_EM_stopCriterion.percentChg);
        }
    }
    
#pragma mark -
#pragma mark File IO
    void write(ostream& outStream, bool writeTrainingSet=false)
    {
        // TODO: write function
    }
    
    void read(istream& inStream, bool readTrainingSet=false)
    {
        // TODO: read function
    }
    
#pragma mark -
#pragma mark Protected Attributes
protected:
    int dimension_gesture;
    int dimension_sound;
    int dimension_total;
    
    vector<float>  exitProbabilities;
    vector<float> LR_transition;
    double sigma;
};

#endif
