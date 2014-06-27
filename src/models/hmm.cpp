//
// hmm.cpp
//
// Hidden Markov Model: Possibly Multimodal and/or submodel of a hierarchical model
//
// Copyright (C) 2014 Ircam - Jules Francoise. All Rights Reserved.
// author: Jules Francoise <jules.francoise@ircam.fr>
//

#include <cmath>
#include "hmm.h"

using namespace std;

#pragma mark -
#pragma mark Constructors
HMM::HMM(rtml_flags flags,
         TrainingSet *trainingSet,
         int nbStates,
         int nbMixtureComponents)
: ProbabilisticModel(flags, trainingSet)
{
    is_hierarchical_ = (flags & HIERARCHICAL);
    
    nbStates_ = nbStates;
    nbMixtureComponents_ = nbMixtureComponents;
    varianceOffset_relative_ = GAUSSIAN_DEFAULT_VARIANCE_OFFSET_RELATIVE;
    varianceOffset_absolute_ = GAUSSIAN_DEFAULT_VARIANCE_OFFSET_ABSOLUTE;
    weight_regression_ = 1.;
    
    allocate();
    
    for (int i=0; i<nbStates; i++) {
        states_[i].set_trainingSet(trainingSet);
    }
    
    play_EM_stopCriterion_.minSteps = PLAY_EM_STEPS;
    play_EM_stopCriterion_.maxSteps = 0;
    play_EM_stopCriterion_.percentChg = PLAY_EM_MAX_LOG_LIK_PERCENT_CHG;
    
    transitionMode_ = LEFT_RIGHT;
    estimateMeans_ = HMM_DEFAULT_ESTIMATEMEANS;
    
    train_EM_init();
}

HMM::HMM(HMM const& src)
{
    _copy(this, src);
}

HMM& HMM::operator=(HMM const& src)
{
    if(this != &src)
    {
        _copy(this, src);
    }
    return *this;
}

void HMM::_copy(HMM *dst,
                HMM const& src)
{
    ProbabilisticModel::_copy(dst, src);
    dst->is_hierarchical_ = src.is_hierarchical_;
    dst->nbMixtureComponents_ = src.nbMixtureComponents_;
    dst->varianceOffset_relative_ = src.varianceOffset_relative_;
    dst->varianceOffset_absolute_ = src.varianceOffset_absolute_;
    dst->weight_regression_ = src.weight_regression_;
    dst->nbStates_ = src.nbStates_;
    dst->estimateMeans_ = src.estimateMeans_;
    
    dst->alpha.resize(dst->nbStates_);
    dst->previousAlpha_.resize(dst->nbStates_);
    dst->beta_.resize(dst->nbStates_);
    dst->previousBeta_.resize(dst->nbStates_);
    
    dst->transition_ = src.transition_;
    dst->prior_ = src.prior_;
    dst->transitionMode_ = src.transitionMode_;
    
    dst->states_ = src.states_;
    dst->play_EM_stopCriterion_ = src.play_EM_stopCriterion_;
}


HMM::~HMM()
{
    prior_.clear();
    transition_.clear();
    alpha.clear();
    previousAlpha_.clear();
    beta_.clear();
    previousBeta_.clear();
    states_.clear();
    if (is_hierarchical_) {
        for (int i=0 ; i<3 ; i++)
            this->alpha_h[i].clear();
        exitProbabilities_.clear();
    }
}

#pragma mark -
#pragma mark Parameters initialization


void HMM::allocate()
{
    prior_.resize(nbStates_);
    transition_.resize(nbStates_*nbStates_);
    alpha.resize(nbStates_);
    previousAlpha_.resize(nbStates_);
    beta_.resize(nbStates_);
    previousBeta_.resize(nbStates_);
    states_.assign(nbStates_, GMM(flags_, this->trainingSet, nbMixtureComponents_, varianceOffset_relative_, varianceOffset_absolute_));
    if (is_hierarchical_)
        updateExitProbabilities(NULL);
}


void HMM::evaluateNbStates(int factor)
{
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    this->set_nbStates(((*this->trainingSet)(0))->second->length() / factor);
}


void HMM::initParametersToDefault()
{
    for (int i=0; i<nbStates_; i++) {
        states_[i].initParametersToDefault();
    }
}


void HMM::initMeansWithFirstPhrase()
{
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    
    for (int n=0; n<nbStates_; n++)
        for (int d=0; d<dimension_; d++)
            states_[n].components[0].mean[d] = 0.0;
    
    vector<int> factor(nbStates_, 0);
    int step = ((*this->trainingSet)(0))->second->length() / nbStates_;
    int offset(0);
    for (int n=0; n<nbStates_; n++) {
        for (int t=0; t<step; t++) {
            for (int d=0; d<dimension_; d++) {
                states_[n].components[0].mean[d] += (*((*this->trainingSet)(0)->second))(offset+t, d);
            }
        }
        offset += step;
        factor[n] += step;
    }
    
    for (int n=0; n<nbStates_; n++)
        for (int d=0; d<dimension_; d++)
            states_[n].components[0].mean[d] /= factor[n];
}


void HMM::initMeansWithAllPhrases_single()
{
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    int nbPhrases = this->trainingSet->size();
    
    for (int n=0; n<nbStates_; n++)
        for (int d=0; d<dimension_; d++)
            states_[n].components[0].mean[d] = 0.0;
    
    vector<int> factor(nbStates_, 0);
    for (int i=0; i<nbPhrases; i++) {
        int step = ((*this->trainingSet)(i))->second->length() / nbStates_;
        int offset(0);
        for (int n=0; n<nbStates_; n++) {
            for (int t=0; t<step; t++) {
                for (int d=0; d<dimension_; d++) {
                    states_[n].components[0].mean[d] += (*((*this->trainingSet)(i)->second))(offset+t, d);
                }
            }
            offset += step;
            factor[n] += step;
        }
    }
    
    for (int n=0; n<nbStates_; n++)
        for (int d=0; d<dimension_; d++)
            states_[n].components[0].mean[d] /= factor[n];
}


void HMM::initCovariancesWithAllPhrases_single()
{
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    int nbPhrases = this->trainingSet->size();
    
    for (int n=0; n<nbStates_; n++)
        for (int d1=0; d1<dimension_; d1++)
            for (int d2=0; d2<dimension_; d2++)
                states_[n].components[0].covariance[d1*dimension_+d2] = -states_[n].components[0].mean[d1]*states_[n].components[0].mean[d2];
    
    vector<int> factor(nbStates_, 0);
    for (int i=0; i<nbPhrases; i++) {
        int step = ((*this->trainingSet)(i))->second->length() / nbStates_;
        int offset(0);
        for (int n=0; n<nbStates_; n++) {
            for (int t=0; t<step; t++) {
                for (int d1=0; d1<dimension_; d1++) {
                    for (int d2=0; d2<dimension_; d2++) {
                        states_[n].components[0].covariance[d1*dimension_+d2] += (*((*this->trainingSet)(i)->second))(offset+t, d1) * (*((*this->trainingSet)(i)->second))(offset+t, d2);
                    }
                }
            }
            offset += step;
            factor[n] += step;
        }
    }
    
    for (int n=0; n<nbStates_; n++)
        for (int d1=0; d1<dimension_; d1++)
            for (int d2=0; d2<dimension_; d2++)
                states_[n].components[0].covariance[d1*dimension_+d2] /= factor[n];
}


void HMM::initMeansWithAllPhrases_mixture()
{
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    int nbPhrases = this->trainingSet->size();
    
    for (int i=0; i<min(nbPhrases, nbMixtureComponents_); i++) {
        int step = ((*this->trainingSet)(i))->second->length() / nbStates_;
        int offset(0);
        for (int n=0; n<nbStates_; n++) {
            for (int d=0; d<dimension_; d++) {
                states_[n].components[i].mean[d] = 0.0;
            }
            for (int t=0; t<step; t++) {
                for (int d=0; d<dimension_; d++) {
                    states_[n].components[i].mean[d] += (*((*this->trainingSet)(i)->second))(offset+t, d) / float(step);
                }
            }
            offset += step;
        }
    }
}


void HMM::initCovariancesWithAllPhrases_mixture()
{
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    int nbPhrases = this->trainingSet->size();
    
    for (int i=0; i<min(nbPhrases, nbMixtureComponents_); i++) {
        int step = ((*this->trainingSet)(i))->second->length() / nbStates_;
        int offset(0);
        for (int n=0; n<nbStates_; n++) {
            for (int d1=0; d1<dimension_; d1++) {
                for (int d2=0; d2<dimension_; d2++) {
                    states_[n].components[i].covariance[d1*dimension_+d2] = -states_[n].components[i].mean[d1]*states_[n].components[i].mean[d2];
                }
            }
            for (int t=0; t<step; t++) {
                for (int d1=0; d1<dimension_; d1++) {
                    for (int d2=0; d2<dimension_; d2++) {
                        states_[n].components[i].covariance[d1*dimension_+d2] += (*((*this->trainingSet)(i)->second))(offset+t, d1) * (*((*this->trainingSet)(i)->second))(offset+t, d2) / float(step);
                    }
                }
            }
            offset += step;
        }
    }
}


void HMM::setErgodic()
{
    for (int i=0 ; i<nbStates_; i++) {
        prior_[i] = 1/(float)nbStates_;
        for (int j=0; j<nbStates_; j++) {
            transition_[i*nbStates_+j] = 1/(float)nbStates_;
        }
    }
}


void HMM::setLeftRight()
{
    for (int i=0 ; i<nbStates_; i++) {
        prior_[i] = 0.;
        for (int j=0; j<nbStates_; j++) {
            transition_[i*nbStates_+j] = ((i == j) || ((i+1) == j)) ? 0.5 : 0;
        }
    }
    transition_[nbStates_*nbStates_-1] = 1.;
    prior_[0] = 1.;
}


void HMM::normalizeTransitions()
{
    double norm_prior(0.), norm_transition;
    for (int i=0; i<nbStates_; i++) {
        norm_prior += prior_[i];
        norm_transition = 0.;
        for (int j=0; j<nbStates_; j++)
            norm_transition += transition_[i*nbStates_+j];
        for (int j=0; j<nbStates_; j++)
            transition_[i*nbStates_+j] /= norm_transition;
    }
    for (int i=0; i<nbStates_; i++)
        prior_[i] /= norm_prior;
}

#pragma mark -
#pragma mark Accessors


int HMM::get_nbStates() const
{
    return nbStates_;
}


void HMM::set_nbStates(int nbStates)
{
    if (nbStates < 1) throw invalid_argument("Number of states must be > 0");;
    if (nbStates == nbStates_) return;
    
    nbStates_ = nbStates;
    allocate();
    
    this->trained = false;
}


int HMM::get_nbMixtureComponents() const
{
    return nbMixtureComponents_;
}


void HMM::set_nbMixtureComponents(int nbMixtureComponents)
{
    if (nbMixtureComponents < 1) throw invalid_argument("The number of Gaussian mixture components must be > 0");;
    if (nbMixtureComponents == nbMixtureComponents_) return;
    
    for (int i=0; i<nbStates_; i++) {
        states_[i].set_nbMixtureComponents(nbMixtureComponents);
    }
    
    nbMixtureComponents_ = nbMixtureComponents;
    
    this->trained = false;
}


double HMM::get_varianceOffset_relative() const
{
    return varianceOffset_relative_;
}


double HMM::get_varianceOffset_absolute() const
{
    return varianceOffset_absolute_;
}


void HMM::set_varianceOffset(double varianceOffset_relative, double varianceOffset_absolute)
{
    for (int i=0; i<nbStates_; i++) {
        states_[i].set_varianceOffset(varianceOffset_relative, varianceOffset_absolute);
    }
    varianceOffset_relative_ = varianceOffset_relative;
    varianceOffset_absolute_ = varianceOffset_absolute;
}

double HMM::get_weight_regression() const
{
    return weight_regression_;
}


void HMM::set_weight_regression(double weight_regression)
{
    weight_regression_ = weight_regression;
    for (int i=0; i<nbStates_; i++) {
        states_[i].set_weight_regression(weight_regression_);
    }
}

string HMM::get_transitionMode() const
{
    if (transitionMode_ == ERGODIC) {
        return "ergodic";
    } else {
        return "left-right";
    }
}


void HMM::set_transitionMode(string transMode_str)
{
    if (!transMode_str.compare("ergodic")) {
        transitionMode_ = ERGODIC;
    } else if (!transMode_str.compare("left-right")) {
        transitionMode_ = LEFT_RIGHT;
    } else {
        throw invalid_argument("Wrong Transition mode. choose 'ergodic' or 'left-right'");
    }
}

#pragma mark -
#pragma mark Forward-Backward algorithm


double HMM::forward_init(const float* observation,
                         const float* observation_output)
{
    double norm_const(0.);
    for (int i=0 ; i<nbStates_ ; i++) {
        if (bimodal_) {
            if (observation_output)
                alpha[i] = prior_[i] * states_[i].obsProb_bimodal(observation, observation_output);
            else
                alpha[i] = prior_[i] * states_[i].obsProb_input(observation);
        } else {
            alpha[i] = prior_[i] * states_[i].obsProb(observation);
        }
        norm_const += alpha[i];
    }
    if (norm_const > 0) {
        for (int i=0 ; i<nbStates_ ; i++) {
            alpha[i] /= norm_const;
        }
        return 1/norm_const;
    } else {
        for (int j=0; j<nbStates_; j++) {
            alpha[j] = 1./double(nbStates_);
        }
        return 1.;
    }
}


double HMM::forward_update(const float* observation,
                           const float* observation_output)
{
    double norm_const(0.);
    previousAlpha_ = alpha;
    for (int j=0; j<nbStates_; j++) {
        alpha[j] = 0.;
        for (int i=0; i<nbStates_; i++) {
            alpha[j] += previousAlpha_[i] * transition_[i*nbStates_+j];
        }
        if (bimodal_) {
            if (observation_output)
                alpha[j] *= states_[j].obsProb_bimodal(observation, observation_output);
            else
                alpha[j] *= states_[j].obsProb_input(observation);
        } else {
            alpha[j] *= states_[j].obsProb(observation);
        }
        norm_const += alpha[j];
    }
    if (norm_const > 1e-300) {
        for (int j=0; j<nbStates_; j++) {
            alpha[j] /= norm_const;
        }
        return 1./norm_const;
    } else {
        return 0.;
        for (int j=0; j<nbStates_; j++) {
            alpha[j] = 1./double(nbStates_);
        }
        return 1.;
    }
}


double HMM::forward_update_withNewObservation(const float* observation,
                                              const float* observation_output)
{
    if (forwardInitialized_) {
        double norm_const(0.);
        for (int j=0; j<nbStates_; j++) {
            alpha[j] = 0.;
            for (int i=0; i<nbStates_; i++) {
                alpha[j] += previousAlpha_[i] * transition_[i*nbStates_+j];
            }
            alpha[j] *= states_[j].obsProb_bimodal(observation, observation_output);
            norm_const += alpha[j];
        }
        if (norm_const > 0) {
            for (int j=0; j<nbStates_; j++) {
                alpha[j] /= norm_const;
            }
            return 1./norm_const;
        } else {
            for (int j=0; j<nbStates_; j++) {
                alpha[j] = 1./double(nbStates_);
            }
            return 1.;
        }
    } else {
        return forward_init(observation, observation_output);
    }
}


void HMM::backward_init(double ct)
{
    for (int i=0 ; i<nbStates_ ; i++)
        beta_[i] = ct;
}


void HMM::backward_update(double ct,
                          const float* observation,
                          const float* observation_output)
{
    previousBeta_ = beta_;
    for (int i=0 ; i<nbStates_; i++) {
        beta_[i] = 0.;
        for (int j=0; j<nbStates_; j++) {
            if (bimodal_) {
                if (observation_output)
                    beta_[i] += transition_[i*nbStates_+j] * previousBeta_[j] * states_[j].obsProb_bimodal(observation, observation_output);
                else
                    beta_[i] += transition_[i*nbStates_+j] * previousBeta_[j] * states_[j].obsProb_input(observation);
            } else {
                beta_[i] += transition_[i*nbStates_+j] * previousBeta_[j] * states_[j].obsProb(observation);
            }
            
        }
        beta_[i] *= ct;
        if (isnan(beta_[i]) || isinf(abs(beta_[i]))) {
            beta_[i] = 1e100;
        }
    }
}

#pragma mark -
#pragma mark Training algorithm


void HMM::train_EM_init()
{
    // Initialize Model Parameters
    // ---------------------------------------
    if (transitionMode_ == ERGODIC) {
        setErgodic();
    } else {
        setLeftRight();
    }
    for (int i=0; i<nbStates_; i++) {
        states_[i].train_EM_init();
    }
    
    if (!this->trainingSet) return;
    
    if (nbMixtureComponents_ > 1) {
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
    gammaSequence_.resize(nbPhrases);
    epsilonSequence_.resize(nbPhrases);
    gammaSequencePerMixture_.resize(nbPhrases);
    int maxT(0);
    int i(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
        int T = it->second->length();
        gammaSequence_[i].resize(T*nbStates_);
        epsilonSequence_[i].resize(T*nbStates_*nbStates_);
        gammaSequencePerMixture_[i].resize(nbMixtureComponents_);
        for (int c=0; c<nbMixtureComponents_; c++) {
            gammaSequencePerMixture_[i][c].resize(T*nbStates_);
        }
        if (T>maxT) {
            maxT = T;
        }
        i++;
    }
    alpha_seq_.resize(maxT*nbStates_);
    beta_seq_.resize(maxT*nbStates_);
    
    gammaSum_.resize(nbStates_);
    gammaSumPerMixture_.resize(nbStates_*nbMixtureComponents_);
}


void HMM::train_EM_terminate()
{
    normalizeTransitions();
    ProbabilisticModel::train_EM_terminate();
}


double HMM::train_EM_update()
{
    double log_prob(0.);
    
    // Forward-backward for each phrase
    // =================================================
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
        log_prob += baumWelch_forwardBackward(it->second, phraseIndex++);
    }
    
    baumWelch_gammaSum();
    
    // Re-estimate model parameters
    // =================================================
    
    // set covariance and mixture coefficients to zero for each state
    for (int i=0; i<nbStates_; i++) {
        for (int c=0; c<nbMixtureComponents_; c++) {
            states_[i].mixtureCoeffs[c] = 0.;
            states_[i].components[c].covariance.assign(dimension_ * dimension_, 0.0);
        }
    }
    
    baumWelch_estimateMixtureCoefficients();
    if (estimateMeans_)
        baumWelch_estimateMeans();
    
    baumWelch_estimateCovariances();
    if (transitionMode_ == ERGODIC)
        baumWelch_estimatePrior();
    baumWelch_estimateTransitions();
    
    return log_prob;
}


double HMM::baumWelch_forwardBackward(Phrase* currentPhrase, int phraseIndex)
{
    int T = currentPhrase->length();
    
    vector<double> ct(T);
    vector<double>::iterator alpha_seq_it = alpha_seq_.begin();
    
    double log_prob;
    
    // Forward algorithm
    if (bimodal_) {
        ct[0] = forward_init(currentPhrase->get_dataPointer_input(0),
                             currentPhrase->get_dataPointer_output(0));
    } else {
        ct[0] = forward_init(currentPhrase->get_dataPointer(0));
    }
    log_prob = -log(ct[0]);
    copy(alpha.begin(), alpha.end(), alpha_seq_it);
    alpha_seq_it += nbStates_;
    
    for (int t=1; t<T; t++) {
        if (bimodal_) {
            ct[t] = forward_update(currentPhrase->get_dataPointer_input(t),
                                   currentPhrase->get_dataPointer_output(t));
        } else {
            ct[t] = forward_update(currentPhrase->get_dataPointer(t));
        }
        log_prob -= log(ct[t]);
        copy(alpha.begin(), alpha.end(), alpha_seq_it);
        alpha_seq_it += nbStates_;
    }
    
    // Backward algorithm
    backward_init(ct[T-1]);
    vector<double>::iterator beta_seq_it = beta_seq_.begin()+(T-1)*nbStates_;
    copy(beta_.begin(), beta_.end(), beta_seq_it);
    beta_seq_it -= nbStates_;
    
    for (int t=T-2; t>=0; t--) {
        if (bimodal_) {
            backward_update(ct[t],
                            currentPhrase->get_dataPointer_input(t+1),
                            currentPhrase->get_dataPointer_output(t+1));
        } else {
            backward_update(ct[t], currentPhrase->get_dataPointer(t+1));
        }
        copy(beta_.begin(), beta_.end(), beta_seq_it);
        beta_seq_it -= nbStates_;
    }
    
    // Compute Gamma Variable
    for (int t=0; t<T; t++) {
        for (int i=0; i<nbStates_; i++) {
            gammaSequence_[phraseIndex][t*nbStates_+i] = alpha_seq_[t*nbStates_+i] * beta_seq_[t*nbStates_+i] / ct[t];
        }
    }
    
    // Compute Gamma variable for each mixture component
    double oo;
    double norm_const;
    
    for (int t=0; t<T; t++) {
        for (int i=0; i<nbStates_; i++) {
            norm_const = 0.;
            for (int c=0; c<nbMixtureComponents_; c++) {
                if (bimodal_) {
                    oo = states_[i].obsProb_bimodal(currentPhrase->get_dataPointer_input(t),
                                                    currentPhrase->get_dataPointer_output(t),
                                                    c);
                } else {
                    oo = states_[i].obsProb(currentPhrase->get_dataPointer(t),
                                            c);
                }
                gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i] = gammaSequence_[phraseIndex][t*nbStates_+i] * oo;
                norm_const += oo;
            }
            if (norm_const > 0)
                for (int c=0; c<nbMixtureComponents_; c++)
                    gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i] /= norm_const;
        }
    }
    
    // Compute Epsilon Variable
    for (int t=0; t<T-1; t++) {
        for (int i=0; i<nbStates_; i++) {
            for (int j=0; j<nbStates_; j++) {
                epsilonSequence_[phraseIndex][t*nbStates_*nbStates_+i*nbStates_+j] = alpha_seq_[t*nbStates_+i]
                * transition_[i*nbStates_+j]
                * beta_seq_[(t+1)*nbStates_+j];
                if (bimodal_) {
                    epsilonSequence_[phraseIndex][t*nbStates_*nbStates_+i*nbStates_+j] *= states_[j].obsProb_bimodal(currentPhrase->get_dataPointer_input(t+1),
                                                                                                                     currentPhrase->get_dataPointer_output(t+1));
                } else {
                    epsilonSequence_[phraseIndex][t*nbStates_*nbStates_+i*nbStates_+j] *= states_[j].obsProb(currentPhrase->get_dataPointer(t+1));
                }
            }
        }
    }
    
    return log_prob;
}

void HMM::baumWelch_gammaSum()
{
    for (int i=0; i<nbStates_; i++) {
        gammaSum_[i] = 0.;
        for (int c=0; c<nbMixtureComponents_; c++) {
            gammaSumPerMixture_[i*nbMixtureComponents_+c] = 0.;
        }
    }
    
    int phraseLength;
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++) {
        phraseLength = it->second->length();
        for (int i=0; i<nbStates_; i++) {
            for (int t=0; t<phraseLength; t++) {
                gammaSum_[i] += gammaSequence_[phraseIndex][t*nbStates_+i];
                for (int c=0; c<nbMixtureComponents_; c++) {
                    gammaSumPerMixture_[i*nbMixtureComponents_+c] += gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i];
                }
            }
        }
        phraseIndex++;
    }
}


void HMM::baumWelch_estimateMixtureCoefficients()
{
    int phraseLength;
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
    {
        phraseLength = it->second->length();
        for (int i=0; i<nbStates_; i++) {
            for (int t=0; t<phraseLength; t++) {
                for (int c=0; c<nbMixtureComponents_; c++) {
                    states_[i].mixtureCoeffs[c] += gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i];
                }
            }
        }
        phraseIndex++;
    }
    
    // Scale mixture coefficients
    for (int i=0; i<nbStates_; i++) {
        states_[i].normalizeMixtureCoeffs();
    }
}


void HMM::baumWelch_estimateMeans()
{
    int phraseLength;
    
    for (int i=0; i<nbStates_; i++) {
        for (int c=0; c<nbMixtureComponents_; c++) {
            states_[i].components[c].mean.assign(dimension_, 0.0);
        }
    }
    
    // Re-estimate Mean
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
    {
        phraseLength = it->second->length();
        for (int i=0; i<nbStates_; i++) {
            for (int t=0; t<phraseLength; t++) {
                for (int c=0; c<nbMixtureComponents_; c++) {
                    for (int d=0; d<dimension_; d++) {
                        states_[i].components[c].mean[d] += gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i] * (*it->second)(t, d);
                    }
                }
            }
        }
        phraseIndex++;
    }
    
    // Normalize mean
    for (int i=0; i<nbStates_; i++) {
        for (int c=0; c<nbMixtureComponents_; c++) {
            for (int d=0; d<dimension_; d++) {
                if (gammaSumPerMixture_[i*nbMixtureComponents_+c] > 0) {
                    states_[i].components[c].mean[d] /= gammaSumPerMixture_[i*nbMixtureComponents_+c];
                }
                if (isnan(states_[i].components[c].mean[d]))
                    throw runtime_error("Convergence Error");
            }
        }
    }
}


void HMM::baumWelch_estimateCovariances()
{
    int phraseLength;
    
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
    {
        phraseLength = it->second->length();
        for (int i=0; i<nbStates_; i++) {
            for (int t=0; t<phraseLength; t++) {
                for (int c=0; c<nbMixtureComponents_; c++) {
                    for (int d1=0; d1<dimension_; d1++) {
                        for (int d2=0; d2<dimension_; d2++) {
                            states_[i].components[c].covariance[d1*dimension_+d2] += gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i]
                            * ((*it->second)(t, d1) - states_[i].components[c].mean[d1])
                            * ((*it->second)(t, d2) - states_[i].components[c].mean[d2]);
                        }
                    }
                }
            }
        }
        phraseIndex++;
    }
    
    // Scale covariance
    for (int i=0; i<nbStates_; i++) {
        for (int c=0; c<nbMixtureComponents_; c++) {
            if (gammaSumPerMixture_[i*nbMixtureComponents_+c] > 0) {
                for (int d=0; d<dimension_*dimension_; d++) {
                    states_[i].components[c].covariance[d] /= gammaSumPerMixture_[i*nbMixtureComponents_+c];
                }
            }
        }
        states_[i].addCovarianceOffset();
        states_[i].updateInverseCovariances();
    }
}


void HMM::baumWelch_estimatePrior()
{
    // Set prior vector to 0
    for (int i=0; i<nbStates_; i++)
        prior_[i] = 0.;
    
    // Re-estimate Prior probabilities
    double sumprior = 0.;
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
    {
        for (int i=0; i<nbStates_; i++) {
            prior_[i] += gammaSequence_[phraseIndex][i];
            sumprior += gammaSequence_[phraseIndex][i];
        }
        phraseIndex++;
    }
    
    // Scale Prior vector
    if (sumprior == 0) {
        cout << "sumprior == 0" << endl;
    }
    for (int i=0; i<nbStates_; i++) {
        prior_[i] /= sumprior;
    }
}


void HMM::baumWelch_estimateTransitions()
{
    // Set prior vector and transition matrix to 0
    for (int i=0; i<nbStates_; i++)
        for (int j=0; j<nbStates_; j++)
            transition_[i*nbStates_+j] = 0.;
    
    int phraseLength;
    // Re-estimate Prior and Transition probabilities
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
    {
        phraseLength = it->second->length();
        for (int i=0; i<nbStates_; i++) {
            for (int j=0; j<nbStates_; j++)
            {
                for (int t=0; t<phraseLength-1; t++) {
                    transition_[i*nbStates_+j] += epsilonSequence_[phraseIndex][t*nbStates_*nbStates_+i*nbStates_+j];
                }
            }
        }
        phraseIndex++;
    }
    
    // Scale transition matrix
    for (int i=0; i<nbStates_; i++) {
        for (int j=0; j<nbStates_; j++) {
            if (gammaSum_[i] > 0)
                transition_[i*nbStates_+j] /= gammaSum_[i];
            if (isnan(transition_[i*nbStates_+j]))
                throw runtime_error("Convergence Error. Check your training data or increase the variance offset");
        }
    }
}

#pragma mark -
#pragma mark Performance


void HMM::performance_init()
{
    ProbabilisticModel::performance_init();
    forwardInitialized_ = false;
    if (is_hierarchical_) {
        for (int i=0 ; i<3 ; i++)
            alpha_h[i].resize(this->nbStates_, 0.0);
        alpha.clear();
        previousAlpha_.clear();
        beta_.clear();
        previousBeta_.clear();
    }
    if (bimodal_)
        results_predicted_output.resize(dimension_ - dimension_input_);
}


void HMM::addCyclicTransition(double proba)
{
    transition_[(nbStates_-1)*nbStates_] = proba; // Add Cyclic Transition probability
}


double HMM::performance_update(vector<float> const& observation)
{
    double ct;
    
    if (forwardInitialized_) {
        ct = forward_update(&observation[0]);
    } else {
        this->likelihoodBuffer_.clear();
        ct = forward_init(&observation[0]);
    }
    
    forwardInitialized_ = true;
    
    if (bimodal_) {
        regression(observation, results_predicted_output);
        
        // Em-like estimation of the output sequence: deprecated now but need to be tested.
        // ========================================================================================
        // double obs_prob(log(ct)), old_obs_prob;
        // int n(1);
        // do
        // {
        //     old_obs_prob = obs_prob;
        //     forward_update_withNewObservation(observation, observation + dimension_input);
        //     regression(observation);
        //     ++n;
        // } while (!play_EM_stop(n, obs_prob, old_obs_prob));
    }
    
    this->updateLikelihoodBuffer(1./ct);
    // TODO: Put this in forward algorithm
    updateTimeProgression();
    
    return results_instant_likelihood;
}

unsigned int argmax(vector<double> const& v)
{
    unsigned int amax(-1);
    if (v.size() == 0)
        return amax;
    double current_max(v[0]);
    for (unsigned int i=1 ; i<v.size() ; ++i) {
        if (v[i] > current_max) {
            current_max = v[i];
            amax = i;
        }
    }
    return amax;
}

void HMM::regression(vector<float> const& observation_input,
                     vector<float>& predicted_output)
{
    int dimension_output = dimension_ - dimension_input_;
    predicted_output.assign(dimension_output, 0.0);
    vector<float> tmp_predicted_output(dimension_output);
    
    for (int i=0; i<nbStates_; i++) {
        states_[i].likelihood(observation_input);
        states_[i].regression(observation_input, tmp_predicted_output);
        for (int d = 0; d < dimension_output; ++d)
        {
            if (is_hierarchical_)
                predicted_output[d] += (alpha_h[0][i] + alpha_h[1][i]) * tmp_predicted_output[d];
            else
                predicted_output[d] += alpha[i] * tmp_predicted_output[d];
        }
    }
}

void HMM::updateTimeProgression()
{
    results_progress = 0.0;
    for (unsigned int i=0 ; i<nbStates_; i++) {
        if (is_hierarchical_)
            results_progress += alpha_h[0][i] * (i+1);
        else
            results_progress += alpha[i] * (i+1);
    }
    results_progress /= double(nbStates_-1);
}

#pragma mark -
#pragma mark File IO


JSONNode HMM::to_json() const
{
    JSONNode json_hmm(JSON_NODE);
    json_hmm.set_name("HMM");
    
    // Write Parent: EM Learning Model
    JSONNode json_emmodel = ProbabilisticModel::to_json();
    json_emmodel.set_name("ProbabilisticModel");
    json_hmm.push_back(json_emmodel);
    
    // Scalar Attributes
    json_hmm.push_back(JSONNode("is_hierarchical", is_hierarchical_));
    json_hmm.push_back(JSONNode("estimatemeans", estimateMeans_));
    json_hmm.push_back(JSONNode("dimension", dimension_));
    json_hmm.push_back(JSONNode("nbstates", nbStates_));
    json_hmm.push_back(JSONNode("nbmixturecomponents", nbMixtureComponents_));
    json_hmm.push_back(JSONNode("varianceoffset_relative", varianceOffset_relative_));
    json_hmm.push_back(JSONNode("varianceoffset_absolute", varianceOffset_absolute_));
    json_hmm.push_back(JSONNode("weight_regression", weight_regression_));
    json_hmm.push_back(JSONNode("transitionmode", int(transitionMode_)));
    
    // Model Parameters
    json_hmm.push_back(vector2json(prior_, "prior"));
    json_hmm.push_back(vector2json(transition_, "transition"));
    if (is_hierarchical_)
        json_hmm.push_back(vector2json(exitProbabilities_, "exitprobabilities"));
    
    // States
    JSONNode json_states(JSON_ARRAY);
    for (int i=0 ; i<nbStates_ ; i++)
    {
        json_states.push_back(states_[i].to_json());
    }
    json_states.set_name("states");
    json_hmm.push_back(json_states);
    
    return json_hmm;
}


void HMM::from_json(JSONNode root)
{
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::iterator root_it = root.begin();
        
        // Get Parent: ProbabilisticModel
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "ProbabilisticModel")
            throw JSONException("Wrong name: was expecting 'ProbabilisticModel'", root_it->name());
        if (root_it->type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root_it->name());
        ProbabilisticModel::from_json(*root_it);
        ++root_it;
        
        // Get If Hierarchical
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "is_hierarchical")
            throw JSONException("Wrong name: was expecting 'is_hierarchical'", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type: was expecting 'JSON_BOOL'", root_it->name());
        if(is_hierarchical_ != root_it->as_bool()) {
            if (is_hierarchical_)
                throw JSONException("Trying to read a non-hierarchical model in a hierarchical model.", root.name());
            else
                throw JSONException("Trying to read a hierarchical model in a non-hierarchical model.", root.name());
        }
        ++root_it;
        
        // Get If estimate means
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "estimatemeans")
            throw JSONException("Wrong name: was expecting 'estimatemeans'", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type: was expecting 'JSON_BOOL'", root_it->name());
        estimateMeans_ = root_it->as_bool();
        ++root_it;
        
        // Get Dimension
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "dimension")
            throw JSONException("Wrong name: was expecting 'dimension'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        dimension_ = root_it->as_int();
        ++root_it;
        
        // Get Number of states
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "nbstates")
            throw JSONException("Wrong name: was expecting 'nbstates'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        nbStates_ = root_it->as_int();
        ++root_it;
        
        // Get Number of Mixture Components
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "nbmixturecomponents")
            throw JSONException("Wrong name: was expecting 'nbmixturecomponents'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        nbMixtureComponents_ = root_it->as_int();
        ++root_it;
        
        // Get Covariance Offset (Relative to data variance)
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "varianceoffset_relative")
            throw JSONException("Wrong name: was expecting 'varianceoffset_relative'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        varianceOffset_relative_ = root_it->as_float();
        ++root_it;
        
        // Get Covariance Offset (Minimum value)
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "varianceoffset_absolute")
            throw JSONException("Wrong name: was expecting 'varianceoffset_absolute'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        varianceOffset_absolute_ = root_it->as_float();
        ++root_it;
        
        // Get Covariance Offset (Minimum value)
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "weight_regression")
            throw JSONException("Wrong name: was expecting 'weight_regression'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        weight_regression_ = root_it->as_float();
        ++root_it;
        
        // Get Transition Mode
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "transitionmode")
            throw JSONException("Wrong name: was expecting 'transitionmode'", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'", root_it->name());
        transitionMode_ = TRANSITION_MODE(root_it->as_int());
        ++root_it;
        
        // Reallocate model parameters
        allocate();
        
        // Get Prior Probabilities
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "prior")
            throw JSONException("Wrong name: was expecting 'prior'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        json2vector(*root_it, prior_, nbStates_);
        ++root_it;
        
        // Get Transition Matrix
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "transition")
            throw JSONException("Wrong name: was expecting 'transition'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        json2vector(*root_it, transition_, nbStates_*nbStates_);
        ++root_it;
        
        // Get Exit probabilities
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "exitprobabilities")
            throw JSONException("Wrong name: was expecting 'exitprobabilities'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        json2vector(*root_it, exitProbabilities_, nbStates_);
        ++root_it;
        
        // Get States
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->name() != "states")
            throw JSONException("Wrong name: was expecting 'states'", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root_it->name());
        for (int i=0 ; i<nbStates_ ; i++) {
            states_[i].from_json((*root_it)[i]);
        }
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (exception &e) {
        throw JSONException(e, root.name());
    }
    
    this->trained = true;
}

#pragma mark -
#pragma mark Exit Probabilities

void HMM::updateExitProbabilities(float *exitProbabilities)
{
    if (!is_hierarchical_)
        throw runtime_error("Model is Not hierarchical: method cannot be used");
    if (exitProbabilities == NULL) {
        exitProbabilities_.resize(this->nbStates_, 0.0);
        exitProbabilities_[this->nbStates_-1] = HMM_DEFAULT_EXITPROBABILITY_LAST_STATE;
    } else {
        exitProbabilities_.resize(this->nbStates_, 0.0);
        for (int i=0 ; i < this->nbStates_ ; i++)
            try {
                exitProbabilities_[i] = exitProbabilities[i];
            } catch (exception &e) {
                throw invalid_argument("Wrong format for exit probabilities");
            }
    }
}


void HMM::addExitPoint(int stateIndex, float proba)
{
    if (!is_hierarchical_)
        throw runtime_error("Model is Not hierarchical: method cannot be used");
    if (stateIndex >= this->nbStates_)
        throw out_of_range("State index out of bounds");
    exitProbabilities_[stateIndex] = proba;
}
