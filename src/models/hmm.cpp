/*
 * hmm.cpp
 *
 * Hidden Markov Model for continuous recognition and regression
 *
 * Contact:
 * - Jules Françoise <jules.francoise@ircam.fr>
 *
 * This code has been initially authored by Jules Françoise
 * <http://julesfrancoise.com> during his PhD thesis, supervised by Frédéric
 * Bevilacqua <href="http://frederic-bevilacqua.net>, in the Sound Music
 * Movement Interaction team <http://ismm.ircam.fr> of the
 * STMS Lab - IRCAM, CNRS, UPMC (2011-2015).
 *
 * Copyright (C) 2015 UPMC, Ircam-Centre Pompidou.
 *
 * This File is part of XMM.
 *
 * XMM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * XMM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with XMM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cmath>
#include "hmm.h"

#pragma mark -
#pragma mark Constructors
xmm::HMM::HMM(xmm_flags flags,
         TrainingSet *trainingSet,
         int nbStates,
         int nbMixtureComponents,
         GaussianDistribution::COVARIANCE_MODE covariance_mode)
: ProbabilisticModel(flags, trainingSet),
nbStates_(nbStates),
nbMixtureComponents_(nbMixtureComponents),
varianceOffset_relative_(GaussianDistribution::DEFAULT_VARIANCE_OFFSET_RELATIVE()),
varianceOffset_absolute_(GaussianDistribution::DEFAULT_VARIANCE_OFFSET_ABSOLUTE()),
covariance_mode_(covariance_mode),
regression_estimator_(FULL),
transitionMode_(LEFT_RIGHT),
estimateMeans_(DEFAULT_ESTIMATEMEANS)
{
    is_hierarchical_ = (flags & HIERARCHICAL);
    allocate();
    initParametersToDefault();
}

xmm::HMM::HMM(HMM const& src)
{
    _copy(this, src);
}

xmm::HMM& xmm::HMM::operator=(HMM const& src)
{
    if(this != &src)
    {
        _copy(this, src);
    }
    return *this;
}

void xmm::HMM::_copy(HMM *dst,
                HMM const& src)
{
    ProbabilisticModel::_copy(dst, src);
    dst->is_hierarchical_ = src.is_hierarchical_;
    dst->nbMixtureComponents_ = src.nbMixtureComponents_;
    dst->varianceOffset_relative_ = src.varianceOffset_relative_;
    dst->varianceOffset_absolute_ = src.varianceOffset_absolute_;
    dst->regression_estimator_ = src.regression_estimator_;
    dst->nbStates_ = src.nbStates_;
    dst->estimateMeans_ = src.estimateMeans_;
    
    dst->alpha.resize(dst->nbStates_);
    dst->previousAlpha_.resize(dst->nbStates_);
    dst->beta_.resize(dst->nbStates_);
    dst->previousBeta_.resize(dst->nbStates_);
    
    dst->transitionMode_ = src.transitionMode_;
    dst->covariance_mode_ = src.covariance_mode_;
    dst->transition = src.transition;
    dst->prior = src.prior;
    dst->exitProbabilities_ = src.exitProbabilities_;
    
    dst->states = src.states;
}


xmm::HMM::~HMM()
{
}

#pragma mark -
#pragma mark Parameters initialization


void xmm::HMM::allocate()
{
    check_training();
    if (transitionMode_ == ERGODIC) {
        prior.resize(nbStates_);
        transition.resize(nbStates_*nbStates_);
    } else {
        prior.clear();
        transition.resize(nbStates_*2);
    }
    alpha.resize(nbStates_);
    previousAlpha_.resize(nbStates_);
    beta_.resize(nbStates_);
    previousBeta_.resize(nbStates_);
    states.assign(nbStates_,
                  GMM(flags_,
                      this->trainingSet,
                      nbMixtureComponents_,
                      varianceOffset_relative_,
                      varianceOffset_absolute_,
                      covariance_mode_));
    if (is_hierarchical_)
        updateExitProbabilities(NULL);
}


void xmm::HMM::evaluateNbStates(int factor)
{
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    this->set_nbStates(((*this->trainingSet)(0))->second->length() / factor);
}


void xmm::HMM::initParametersToDefault()
{
    if (transitionMode_ == ERGODIC) {
        setErgodic();
    } else {
        setLeftRight();
    }
    for (int i=0; i<nbStates_; i++) {
        states[i].initParametersToDefault();
    }
}


void xmm::HMM::initMeansWithAllPhrases()
{
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    int nbPhrases = this->trainingSet->size();
    
    for (int n=0; n<nbStates_; n++)
        for (int d=0; d<dimension_; d++)
            states[n].components[0].mean[d] = 0.0;
    
    std::vector<int> factor(nbStates_, 0);
    for (int i=0; i<nbPhrases; i++) {
        int step = ((*this->trainingSet)(i))->second->length() / nbStates_;
        int offset(0);
        for (int n=0; n<nbStates_; n++) {
            for (int t=0; t<step; t++) {
                for (int d=0; d<dimension_; d++) {
                    states[n].components[0].mean[d] += (*((*this->trainingSet)(i)->second))(offset+t, d);
                }
            }
            offset += step;
            factor[n] += step;
        }
    }
    
    for (int n=0; n<nbStates_; n++)
        for (int d=0; d<dimension_; d++)
            states[n].components[0].mean[d] /= factor[n];
}


void xmm::HMM::initCovariances_fullyObserved()
{
    // TODO: simplify with covariance symmetricity.
    if (!this->trainingSet || this->trainingSet->is_empty()) return;
    int nbPhrases = this->trainingSet->size();
    
    if (covariance_mode_ == GaussianDistribution::FULL) {
        for (int n=0; n<nbStates_; n++)
            states[n].components[0].covariance.assign(dimension_*dimension_, 0.0);
    } else {
        for (int n=0; n<nbStates_; n++)
            states[n].components[0].covariance.assign(dimension_, 0.0);
    }
    
    std::vector<int> factor(nbStates_, 0);
    std::vector<double> othermeans(nbStates_*dimension_, 0.0);
    for (int i=0; i<nbPhrases; i++) {
        int step = ((*this->trainingSet)(i))->second->length() / nbStates_;
        int offset(0);
        for (int n=0; n<nbStates_; n++) {
            for (int t=0; t<step; t++) {
                for (int d1=0; d1<dimension_; d1++) {
                    othermeans[n*dimension_+d1] += (*((*this->trainingSet)(i)->second))(offset+t, d1);
                    if (covariance_mode_ == GaussianDistribution::FULL) {
                        for (int d2=0; d2<dimension_; d2++) {
                            states[n].components[0].covariance[d1*dimension_+d2] += (*((*this->trainingSet)(i)->second))(offset+t, d1) * (*((*this->trainingSet)(i)->second))(offset+t, d2);
                        }
                    } else {
                        float value = (*((*this->trainingSet)(i)->second))(offset+t, d1);
                        states[n].components[0].covariance[d1] += value * value;
                    }
                }
            }
            offset += step;
            factor[n] += step;
        }
    }
    
    for (int n=0; n<nbStates_; n++)
        for (int d1=0; d1<dimension_; d1++) {
            othermeans[n*dimension_+d1] /= factor[n];
            if (covariance_mode_ == GaussianDistribution::FULL) {
                for (int d2=0; d2<dimension_; d2++)
                    states[n].components[0].covariance[d1*dimension_+d2] /= factor[n];
            } else {
                states[n].components[0].covariance[d1] /= factor[n];
            }
        }
    
    for (int n=0; n<nbStates_; n++) {
        for (int d1=0; d1<dimension_; d1++) {
            if (covariance_mode_ == GaussianDistribution::FULL) {
                for (int d2=0; d2<dimension_; d2++)
                    states[n].components[0].covariance[d1*dimension_+d2] -= othermeans[n*dimension_+d1]*othermeans[n*dimension_+d2];
            } else {
                states[n].components[0].covariance[d1] -= othermeans[n*dimension_+d1] * othermeans[n*dimension_+d1];
            }
        }
        states[n].addCovarianceOffset();
        states[n].updateInverseCovariances();
    }
}

void xmm::HMM::initMeansCovariancesWithGMMEM()
{
    int nbPhrases = this->trainingSet->size();
    for (unsigned int n=0; n<nbStates_; n++) {
        TrainingSet temp_ts(SHARED_MEMORY | (bimodal_ ? BIMODAL : NONE), dimension_, dimension_input_);
        for (int i=0; i<nbPhrases; i++) {
            int step = ((*this->trainingSet)(i))->second->length() / nbStates_;
            if (bimodal_)
                temp_ts.connect(i,
                                ((*this->trainingSet)(i))->second->get_dataPointer_input(n*step),
                                ((*this->trainingSet)(i))->second->get_dataPointer_output(n*step),
                                step);
            else
                temp_ts.connect(i,
                                ((*this->trainingSet)(i))->second->get_dataPointer(n*step),
                                step);
        }
        GMM temp_gmm((bimodal_ ? BIMODAL : NONE), &temp_ts, nbMixtureComponents_, varianceOffset_relative_, varianceOffset_absolute_, covariance_mode_);
        temp_gmm.train();
        for (unsigned int c=0; c<nbMixtureComponents_; c++) {
            states[n].components[c].mean = temp_gmm.components[c].mean;
            states[n].components[c].covariance = temp_gmm.components[c].covariance;
            states[n].updateInverseCovariances();
        }
    }
}

void xmm::HMM::setErgodic()
{
    for (int i=0 ; i<nbStates_; i++) {
        prior[i] = 1/(float)nbStates_;
        for (int j=0; j<nbStates_; j++) {
            transition[i*nbStates_+j] = 1/(float)nbStates_;
        }
    }
}


void xmm::HMM::setLeftRight()
{
    transition.assign(nbStates_*2, 0.5);
    transition[(nbStates_-1)*2] = 1.;
    transition[(nbStates_-1)*2+1] = 0.;
}


void xmm::HMM::normalizeTransitions()
{
    double norm_transition;
    if (transitionMode_ == ERGODIC) {
        double norm_prior(0.);
        for (int i=0; i<nbStates_; i++) {
            norm_prior += prior[i];
            norm_transition = 0.;
            for (int j=0; j<nbStates_; j++)
                norm_transition += transition[i*nbStates_+j];
            for (int j=0; j<nbStates_; j++)
                transition[i*nbStates_+j] /= norm_transition;
        }
        for (int i=0; i<nbStates_; i++)
            prior[i] /= norm_prior;
    } else {
        for (int i=0; i<nbStates_; i++) {
            norm_transition = transition[i*2] + transition[i*2+1];
            transition[i*2] /= norm_transition;
            transition[i*2+1] /= norm_transition;
        }
    }
}

#pragma mark -
#pragma mark Accessors
void xmm::HMM::set_trainingSet(TrainingSet *trainingSet)
{
    prevent_attribute_change();
    for (int i=0; i<nbStates_; i++) {
        states[i].set_trainingSet(trainingSet);
    }
    ProbabilisticModel::set_trainingSet(trainingSet);
}

int xmm::HMM::get_nbStates() const
{
    return nbStates_;
}


void xmm::HMM::set_nbStates(int nbStates)
{
    prevent_attribute_change();
    if (nbStates < 1) throw std::invalid_argument("Number of states must be > 0");;
    if (nbStates == nbStates_) return;
    
    nbStates_ = nbStates;
    allocate();
    
    this->trained = false;
}


int xmm::HMM::get_nbMixtureComponents() const
{
    return nbMixtureComponents_;
}


void xmm::HMM::set_nbMixtureComponents(int nbMixtureComponents)
{
    prevent_attribute_change();
    if (nbMixtureComponents < 1) throw std::invalid_argument("The number of Gaussian mixture components must be > 0");;
    if (nbMixtureComponents == nbMixtureComponents_) return;
    
    for (int i=0; i<nbStates_; i++) {
        states[i].set_nbMixtureComponents(nbMixtureComponents);
    }
    
    nbMixtureComponents_ = nbMixtureComponents;
    
    this->trained = false;
}


double xmm::HMM::get_varianceOffset_relative() const
{
    return varianceOffset_relative_;
}


double xmm::HMM::get_varianceOffset_absolute() const
{
    return varianceOffset_absolute_;
}


void xmm::HMM::set_varianceOffset(double varianceOffset_relative, double varianceOffset_absolute)
{
    prevent_attribute_change();
    for (int i=0; i<nbStates_; i++) {
        states[i].set_varianceOffset(varianceOffset_relative, varianceOffset_absolute);
    }
    varianceOffset_relative_ = varianceOffset_relative;
    varianceOffset_absolute_ = varianceOffset_absolute;
}

xmm::GaussianDistribution::COVARIANCE_MODE xmm::HMM::get_covariance_mode() const
{
    return covariance_mode_;
}

void xmm::HMM::set_covariance_mode(GaussianDistribution::COVARIANCE_MODE covariance_mode)
{
    prevent_attribute_change();
    if (covariance_mode == covariance_mode_) return;
    covariance_mode_ = covariance_mode;
    for (unsigned int i=0; i<nbStates_; ++i) {
        states[i].set_covariance_mode(covariance_mode);
    }
}

xmm::HMM::REGRESSION_ESTIMATOR xmm::HMM::get_regression_estimator() const
{
    return regression_estimator_;
}


void xmm::HMM::set_regression_estimator(xmm::HMM::REGRESSION_ESTIMATOR regression_estimator)
{
    regression_estimator_ = regression_estimator;
}

std::string xmm::HMM::get_transitionMode() const
{
    if (transitionMode_ == ERGODIC) {
        return "ergodic";
    } else {
        return "left-right";
    }
}


void xmm::HMM::set_transitionMode(std::string transMode_str)
{
    prevent_attribute_change();
    if (!transMode_str.compare("ergodic")) {
        if (transitionMode_ == ERGODIC)
            return;
        transitionMode_ = ERGODIC;
    } else if (!transMode_str.compare("left-right")) {
        if (transitionMode_ == LEFT_RIGHT)
            return;
        transitionMode_ = LEFT_RIGHT;
    } else {
        throw std::invalid_argument("Wrong Transition mode. choose 'ergodic' or 'left-right'");
    }
    allocate();
}

#pragma mark -
#pragma mark Forward-Backward algorithm
double xmm::HMM::forward_init(const float* observation,
                         const float* observation_output)
{
    double norm_const(0.);
    if (transitionMode_ == ERGODIC) {
        for (int i=0 ; i<nbStates_ ; i++) {
            if (bimodal_) {
                if (observation_output)
                    alpha[i] = prior[i] * states[i].obsProb_bimodal(observation, observation_output);
                else
                    alpha[i] = prior[i] * states[i].obsProb_input(observation);
            } else {
                alpha[i] = prior[i] * states[i].obsProb(observation);
            }
            norm_const += alpha[i];
        }
    } else {
        alpha.assign(nbStates_, 0.0);
        if (bimodal_) {
            if (observation_output)
                alpha[0] = states[0].obsProb_bimodal(observation, observation_output);
            else
                alpha[0] = states[0].obsProb_input(observation);
        } else {
            alpha[0] = states[0].obsProb(observation);
        }
        norm_const += alpha[0];
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


double xmm::HMM::forward_update(const float* observation,
                           const float* observation_output)
{
    double norm_const(0.);
    previousAlpha_ = alpha;
    for (int j=0; j<nbStates_; j++) {
        alpha[j] = 0.;
        if (transitionMode_ == ERGODIC) {
            for (int i=0; i<nbStates_; i++) {
                alpha[j] += previousAlpha_[i] * transition[i*nbStates_+j];
            }
        } else {
            alpha[j] += previousAlpha_[j] * transition[j*2];
            if (j>0) {
                alpha[j] += previousAlpha_[j-1] * transition[(j-1)*2+1];
            } else {
                alpha[0] += previousAlpha_[nbStates_-1] * transition[nbStates_*2-1];
            }
        }
        if (bimodal_) {
            if (observation_output)
                alpha[j] *= states[j].obsProb_bimodal(observation, observation_output);
            else
                alpha[j] *= states[j].obsProb_input(observation);
        } else {
            alpha[j] *= states[j].obsProb(observation);
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
    }
}

void xmm::HMM::backward_init(double ct)
{
    for (int i=0 ; i<nbStates_ ; i++)
        beta_[i] = ct;
}


void xmm::HMM::backward_update(double ct,
                          const float* observation,
                          const float* observation_output)
{
    previousBeta_ = beta_;
    for (int i=0 ; i<nbStates_; i++) {
        beta_[i] = 0.;
        if (transitionMode_ == ERGODIC) {
            for (int j=0; j<nbStates_; j++) {
                if (bimodal_) {
                    if (observation_output)
                        beta_[i] += transition[i*nbStates_+j] * previousBeta_[j] * states[j].obsProb_bimodal(observation, observation_output);
                    else
                        beta_[i] += transition[i*nbStates_+j] * previousBeta_[j] * states[j].obsProb_input(observation);
                } else {
                    beta_[i] += transition[i*nbStates_+j] * previousBeta_[j] * states[j].obsProb(observation);
                }
                
            }
        } else {
            if (bimodal_) {
                if (observation_output)
                    beta_[i] += transition[i*2] * previousBeta_[i] * states[i].obsProb_bimodal(observation, observation_output);
                else
                    beta_[i] += transition[i*2] * previousBeta_[i] * states[i].obsProb_input(observation);
            } else {
                beta_[i] += transition[i*2] * previousBeta_[i] * states[i].obsProb(observation);
            }
            if (i<nbStates_-1) {
                if (bimodal_) {
                    if (observation_output)
                        beta_[i] += transition[i*2+1] * previousBeta_[i+1] * states[i+1].obsProb_bimodal(observation, observation_output);
                    else
                        beta_[i] += transition[i*2+1] * previousBeta_[i+1] * states[i+1].obsProb_input(observation);
                } else {
                    beta_[i] += transition[i*2+1] * previousBeta_[i+1] * states[i+1].obsProb(observation);
                }
            }
        }
        beta_[i] *= ct;
        if (std::isnan(beta_[i]) || std::isinf(fabs(beta_[i]))) {
            beta_[i] = 1e100;
        }
    }
}

#pragma mark -
#pragma mark Training algorithm
void xmm::HMM::train_EM_init()
{
    initParametersToDefault();
    if (!this->trainingSet || this->trainingSet->size() == 0)
        return;
    
    if (nbMixtureComponents_ > 0) { // TODO: weird > 0
        initMeansCovariancesWithGMMEM();
    } else {
        initMeansWithAllPhrases();
        initCovariances_fullyObserved();
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
        if (transitionMode_ == ERGODIC) {
            epsilonSequence_[i].resize(T*nbStates_*nbStates_);
        } else {
            epsilonSequence_[i].resize(T*2*nbStates_);
        }
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


void xmm::HMM::train_EM_terminate()
{
    normalizeTransitions();
    gammaSequence_.clear();
    epsilonSequence_.clear();
    gammaSequencePerMixture_.clear();
    alpha_seq_.clear();
    beta_seq_.clear();
    gammaSum_.clear();
    gammaSumPerMixture_.clear();
    ProbabilisticModel::train_EM_terminate();
}


double xmm::HMM::train_EM_update()
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
            states[i].mixtureCoeffs[c] = 0.;
            if (covariance_mode_ == GaussianDistribution::FULL) {
                states[i].components[c].covariance.assign(dimension_ * dimension_, 0.0);
            } else {
                states[i].components[c].covariance.assign(dimension_, 0.0);
            }
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



double xmm::HMM::baumWelch_forward_update(std::vector<double>::iterator observation_likelihoods)
{
    double norm_const(0.);
    previousAlpha_ = alpha;
    for (int j=0; j<nbStates_; j++) {
        alpha[j] = 0.;
        if (transitionMode_ == ERGODIC) {
            for (int i=0; i<nbStates_; i++) {
                alpha[j] += previousAlpha_[i] * transition[i*nbStates_+j];
            }
        } else {
            alpha[j] += previousAlpha_[j] * transition[j*2];
            if (j>0) {
                alpha[j] += previousAlpha_[j-1] * transition[(j-1)*2+1];
            } else {
                alpha[0] += previousAlpha_[nbStates_-1] * transition[nbStates_*2-1];
            }
        }
        alpha[j] *= observation_likelihoods[j];
        norm_const += alpha[j];
    }
    if (norm_const > 1e-300) {
        for (int j=0; j<nbStates_; j++) {
            alpha[j] /= norm_const;
        }
        return 1./norm_const;
    } else {
        return 0.;
    }
}


void xmm::HMM::baumWelch_backward_update(double ct, std::vector<double>::iterator observation_likelihoods)
{
    previousBeta_ = beta_;
    for (int i=0 ; i<nbStates_; i++) {
        beta_[i] = 0.;
        if (transitionMode_ == ERGODIC) {
            for (int j=0; j<nbStates_; j++) {
                beta_[i] += transition[i*nbStates_+j] * previousBeta_[j] * observation_likelihoods[j];
            }
        } else {
            beta_[i] += transition[i*2] * previousBeta_[i] * observation_likelihoods[i];
            if (i<nbStates_-1) {
                beta_[i] += transition[i*2+1] * previousBeta_[i+1] * observation_likelihoods[i+1];
            }
        }
        beta_[i] *= ct;
        if (std::isnan(beta_[i]) || std::isinf(fabs(beta_[i]))) {
            beta_[i] = 1e100;
        }
    }
}

double xmm::HMM::baumWelch_forwardBackward(Phrase* currentPhrase, int phraseIndex)
{
    int T = currentPhrase->length();
    
    std::vector<double> ct(T);
    std::vector<double>::iterator alpha_seq_it = alpha_seq_.begin();
    
    double log_prob;
    
    std::vector<double> observation_probabilities(nbStates_ * T);
    for (unsigned int t=0; t<T; ++t) {
        for (unsigned int i=0; i<nbStates_; i++) {
            if (bimodal_) {
                observation_probabilities[t * nbStates_ + i] = states[i].obsProb_bimodal(currentPhrase->get_dataPointer_input(t),
                                                                                         currentPhrase->get_dataPointer_output(t));
            } else {
                observation_probabilities[t * nbStates_ + i] = states[i].obsProb(currentPhrase->get_dataPointer(t));
            }
        }
    }
    
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
        ct[t] = baumWelch_forward_update(observation_probabilities.begin() + t * nbStates_);
        log_prob -= log(ct[t]);
        copy(alpha.begin(), alpha.end(), alpha_seq_it);
        alpha_seq_it += nbStates_;
    }
    
    // Backward algorithm
    backward_init(ct[T-1]);
    copy(beta_.begin(), beta_.end(), beta_seq_.begin() + (T - 1)*nbStates_);
    
    for (int t=T-2; t>=0; t--) {
        baumWelch_backward_update(ct[t], observation_probabilities.begin() + (t+1) * nbStates_);
        copy(beta_.begin(), beta_.end(), beta_seq_.begin() + t * nbStates_);
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
            if (nbMixtureComponents_ == 1) {
                oo = observation_probabilities[t * nbStates_ + i];
                gammaSequencePerMixture_[phraseIndex][0][t*nbStates_+i] = gammaSequence_[phraseIndex][t*nbStates_+i] * oo;
                norm_const += oo;
            } else {
                for (int c=0; c<nbMixtureComponents_; c++) {
                    if (bimodal_) {
                        oo = states[i].obsProb_bimodal(currentPhrase->get_dataPointer_input(t),
                                                       currentPhrase->get_dataPointer_output(t),
                                                       c);
                    } else {
                        oo = states[i].obsProb(currentPhrase->get_dataPointer(t),
                                               c);
                    }
                    gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i] = gammaSequence_[phraseIndex][t*nbStates_+i] * oo;
                    norm_const += oo;
                }
            }
            if (norm_const > 0)
                for (int c=0; c<nbMixtureComponents_; c++)
                    gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i] /= norm_const;
        }
    }
    
    // Compute Epsilon Variable
    if (transitionMode_ == ERGODIC) {
        for (int t=0; t<T-1; t++) {
            for (int i=0; i<nbStates_; i++) {
                for (int j=0; j<nbStates_; j++) {
                    epsilonSequence_[phraseIndex][t*nbStates_*nbStates_+i*nbStates_+j] = alpha_seq_[t*nbStates_+i]
                    * transition[i*nbStates_+j]
                    * beta_seq_[(t+1)*nbStates_+j];
                    epsilonSequence_[phraseIndex][t*nbStates_*nbStates_+i*nbStates_+j] *= observation_probabilities[(t+1) * nbStates_ + j];
                }
            }
        }
    } else {
        for (int t=0; t<T-1; t++) {
            for (int i=0; i<nbStates_; i++) {
                epsilonSequence_[phraseIndex][t*2*nbStates_+i*2] = alpha_seq_[t*nbStates_+i]
                    * transition[i*2]
                    * beta_seq_[(t+1)*nbStates_+i];
                epsilonSequence_[phraseIndex][t*2*nbStates_+i*2] *= observation_probabilities[(t+1) * nbStates_ + i];
                if (i<nbStates_-1) {
                    epsilonSequence_[phraseIndex][t*2*nbStates_+i*2+1] = alpha_seq_[t*nbStates_+i]
                        * transition[i*2+1]
                        * beta_seq_[(t+1)*nbStates_+i+1];
                    epsilonSequence_[phraseIndex][t*2*nbStates_+i*2+1] *= observation_probabilities[(t+1) * nbStates_ + i+1];
                }
            }
        }
    }
    
    return log_prob;
}

void xmm::HMM::baumWelch_gammaSum()
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


void xmm::HMM::baumWelch_estimateMixtureCoefficients()
{
    int phraseLength;
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
    {
        phraseLength = it->second->length();
        for (int i=0; i<nbStates_; i++) {
            for (int t=0; t<phraseLength; t++) {
                for (int c=0; c<nbMixtureComponents_; c++) {
                    states[i].mixtureCoeffs[c] += gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i];
                }
            }
        }
        phraseIndex++;
    }
    
    // Scale mixture coefficients
    for (int i=0; i<nbStates_; i++) {
        states[i].normalizeMixtureCoeffs();
    }
}


void xmm::HMM::baumWelch_estimateMeans()
{
    int phraseLength;
    
    for (int i=0; i<nbStates_; i++) {
        for (int c=0; c<nbMixtureComponents_; c++) {
            states[i].components[c].mean.assign(dimension_, 0.0);
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
                        states[i].components[c].mean[d] += gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i] * (*it->second)(t, d);
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
                    states[i].components[c].mean[d] /= gammaSumPerMixture_[i*nbMixtureComponents_+c];
                }
                if (std::isnan(states[i].components[c].mean[d]))
                    throw std::runtime_error("Convergence Error");
            }
        }
    }
}


void xmm::HMM::baumWelch_estimateCovariances()
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
                        if (covariance_mode_ == GaussianDistribution::FULL) {
                            for (int d2=d1; d2<dimension_; d2++) {
                                states[i].components[c].covariance[d1*dimension_+d2] += gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i]
                                * ((*it->second)(t, d1) - states[i].components[c].mean[d1])
                                * ((*it->second)(t, d2) - states[i].components[c].mean[d2]);
                            }
                        } else {
                            float value = (*it->second)(t, d1) - states[i].components[c].mean[d1];
                            states[i].components[c].covariance[d1] += gammaSequencePerMixture_[phraseIndex][c][t*nbStates_+i]
                            * value * value;
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
                for (int d1=0; d1<dimension_; d1++) {
                    if (covariance_mode_ == GaussianDistribution::FULL) {
                        for (int d2=d1; d2<dimension_; d2++) {
                            states[i].components[c].covariance[d1*dimension_+d2] /= gammaSumPerMixture_[i*nbMixtureComponents_+c];
                            if (d1 != d2)
                                states[i].components[c].covariance[d2*dimension_+d1] = states[i].components[c].covariance[d1*dimension_+d2];
                        }
                    } else {
                        states[i].components[c].covariance[d1] /= gammaSumPerMixture_[i*nbMixtureComponents_+c];
                    }
                }
            }
        }
        states[i].addCovarianceOffset();
        states[i].updateInverseCovariances();
    }
}


void xmm::HMM::baumWelch_estimatePrior()
{
    // Set prior vector to 0
    for (int i=0; i<nbStates_; i++)
        prior[i] = 0.;
    
    // Re-estimate Prior probabilities
    double sumprior = 0.;
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
    {
        for (int i=0; i<nbStates_; i++) {
            prior[i] += gammaSequence_[phraseIndex][i];
            sumprior += gammaSequence_[phraseIndex][i];
        }
        phraseIndex++;
    }
    
    // Scale Prior vector
    if (sumprior > 0.) {
        for (int i=0; i<nbStates_; i++) {
            prior[i] /= sumprior;
        }
    }
}


void xmm::HMM::baumWelch_estimateTransitions()
{
    // Set prior vector and transition matrix to 0
    if (transitionMode_ == ERGODIC) {
        transition.assign(nbStates_*nbStates_, 0.0);
    } else {
        transition.assign(nbStates_*2, 0.0);
    }
    
    int phraseLength;
    // Re-estimate Prior and Transition probabilities
    int phraseIndex(0);
    for (phrase_iterator it = this->trainingSet->begin(); it != this->trainingSet->end(); it++)
    {
        phraseLength = it->second->length();
        for (int i=0; i<nbStates_; i++) {
            // Experimental: A bit of regularization (sometimes avoids numerical errors)
            if (transitionMode_ == LEFT_RIGHT) {
                transition[i*2] += TRANSITION_REGULARIZATION();
                if (i<nbStates_-1)
                    transition[i*2+1] += TRANSITION_REGULARIZATION();
                else
                    transition[i*2] += TRANSITION_REGULARIZATION();
            }
            // End Regularization
            if (transitionMode_ == ERGODIC) {
                for (int j=0; j<nbStates_; j++)
                {
                    for (int t=0; t<phraseLength-1; t++) {
                        transition[i*nbStates_+j] += epsilonSequence_[phraseIndex][t*nbStates_*nbStates_+i*nbStates_+j];
                    }
                }
            } else {
                for (int t=0; t<phraseLength-1; t++) {
                    transition[i*2] += epsilonSequence_[phraseIndex][t*2*nbStates_+i*2];
                }
                if (i<nbStates_-1) {
                    for (int t=0; t<phraseLength-1; t++) {
                        transition[i*2+1] += epsilonSequence_[phraseIndex][t*2*nbStates_+i*2+1];
                    }
                }
            }
        }
        phraseIndex++;
    }
    
    // Scale transition matrix
    if (transitionMode_ == ERGODIC) {
        for (int i=0; i<nbStates_; i++) {
            for (int j=0; j<nbStates_; j++) {
                transition[i*nbStates_+j] /= (gammaSum_[i] + 2.*TRANSITION_REGULARIZATION());
                if (std::isnan(transition[i*nbStates_+j]))
                    throw std::runtime_error("Convergence Error. Check your training data or increase the variance offset");
            }
        }
    } else {
        for (int i=0; i<nbStates_; i++) {
            transition[i*2] /= (gammaSum_[i] + 2.*TRANSITION_REGULARIZATION());
            if (std::isnan(transition[i*2]))
                throw std::runtime_error("Convergence Error. Check your training data or increase the variance offset");
            if (i<nbStates_-1) {
                transition[i*2+1] /= (gammaSum_[i] + 2.*TRANSITION_REGULARIZATION());
                if (std::isnan(transition[i*2+1]))
                    throw std::runtime_error("Convergence Error. Check your training data or increase the variance offset");
            }
        }
    }
}

#pragma mark -
#pragma mark Performance


void xmm::HMM::performance_init()
{
    check_training();
    ProbabilisticModel::performance_init();
    forwardInitialized_ = false;
    if (is_hierarchical_) {
        for (int i=0 ; i<3 ; i++)
            alpha_h[i].resize(this->nbStates_, 0.0);
        alpha.clear();
        previousAlpha_.clear();
        beta_.clear();
        previousBeta_.clear();
    } else {
        addCyclicTransition(0.05);
    }
}


void xmm::HMM::addCyclicTransition(double proba)
{
    check_training();
    if (transitionMode_ == ERGODIC) {
        if (nbStates_ > 1)
            transition[(nbStates_-1)*nbStates_] = proba;
    } else {
        if (nbStates_ > 1)
            transition[(nbStates_-1)*2+1] = proba;
    }
}


double xmm::HMM::performance_update(std::vector<float> const& observation)
{
    check_training();
    double ct;
    
    if (forwardInitialized_) {
        ct = forward_update(&observation[0]);
    } else {
        this->likelihoodBuffer_.clear();
        ct = forward_init(&observation[0]);
    }
    
    forwardInitialized_ = true;
    
    this->updateLikelihoodBuffer(1./ct);
    updateAlphaWindow();
    updateTimeProgression();
    
    if (bimodal_) {
        regression(observation, results_predicted_output);
    }
    
    return results_instant_likelihood;
}

unsigned int argmax(std::vector<double> const& v)
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

void xmm::HMM::updateAlphaWindow()
{
    check_training();
    results_likeliest_state = 0;
    // Get likeliest State
    double best_alpha(is_hierarchical_ ? (alpha_h[0][0] + alpha_h[1][0]) : alpha[0]);
    for (unsigned int i = 1; i < nbStates_ ; ++i) {
        if (is_hierarchical_) {
            if ((alpha_h[0][i] + alpha_h[1][i]) > best_alpha) {
                best_alpha = alpha_h[0][i] + alpha_h[1][i];
                results_likeliest_state = i;
            }
        } else {
            if (alpha[i] > best_alpha) {
                best_alpha = alpha[i];
                results_likeliest_state = i;
            }
        }
    }
    
    // Compute Window
    results_window_minindex = (results_likeliest_state - (nbStates_/2));
    results_window_maxindex = (results_likeliest_state + (nbStates_/2));
    results_window_minindex = (results_window_minindex >= 0) ? results_window_minindex : 0;
    results_window_maxindex = (results_window_maxindex <= nbStates_) ? results_window_maxindex : nbStates_;
    results_window_normalization_constant = 0.0;
    for (int i=results_window_minindex; i<results_window_maxindex; ++i) {
        results_window_normalization_constant += is_hierarchical_ ? (alpha_h[0][i] + alpha_h[1][i]) : alpha[i];
    }
}

void xmm::HMM::regression(std::vector<float> const& observation_input,
                          std::vector<float>& predicted_output)
{
    check_training();
    int dimension_output = dimension_ - dimension_input_;
    predicted_output.assign(dimension_output, 0.0);
    results_output_variance.assign(dimension_output, 0.0);
    std::vector<float> tmp_predicted_output(dimension_output);
    
    if (regression_estimator_ == LIKELIEST) {
        states[results_likeliest_state].likelihood(observation_input);
        states[results_likeliest_state].regression(observation_input, predicted_output);
        return;
    }
    
    int clip_min_state = (regression_estimator_ == FULL) ? 0 : results_window_minindex;
    int clip_max_state = (regression_estimator_ == FULL) ? nbStates_ : results_window_maxindex;
    double normalization_constant = (regression_estimator_ == FULL) ? 1.0 : results_window_normalization_constant;
    
    if (normalization_constant <= 0.0) normalization_constant = 1.;
    
    // Compute Regression
    for (int i=clip_min_state; i<clip_max_state; ++i) {
        states[i].likelihood(observation_input);
        states[i].regression(observation_input, tmp_predicted_output);
        for (int d = 0; d < dimension_output; ++d)
        {
            if (is_hierarchical_) {
                predicted_output[d] += (alpha_h[0][i] + alpha_h[1][i]) * tmp_predicted_output[d] / normalization_constant;
                results_output_variance[d] += (alpha_h[0][i] + alpha_h[1][i]) * (alpha_h[0][i] + alpha_h[1][i]) * states[i].results_output_variance[d] / normalization_constant;
            } else {
                predicted_output[d] += alpha[i] * tmp_predicted_output[d] / normalization_constant;
                results_output_variance[d] += alpha[i] * alpha[i] * states[i].results_output_variance[d] / normalization_constant;
            }
        }
    }
}

void xmm::HMM::updateTimeProgression()
{
    results_progress = 0.0;
    for (int i=results_window_minindex; i<results_window_maxindex; ++i) {
        if (is_hierarchical_)
            results_progress += (alpha_h[0][i] + alpha_h[1][i] + alpha_h[2][i]) * i / results_window_normalization_constant;
        else
            results_progress += alpha[i] * i / results_window_normalization_constant;
    }
    results_progress /= double(nbStates_-1);
    
    //    /////////////////////////
    //    results_progress = 0.0;
    //    for (unsigned int i=0 ; i<nbStates_; i++) {
    //        if (is_hierarchical_)
    //            results_progress += (alpha_h[0][i] + alpha_h[1][i] + alpha_h[2][i]) * i;
    //        else
    //            results_progress += alpha[i] * i;
    //    }
    //    results_progress /= double(nbStates_-1);
    //    /////////////////////////
}

#pragma mark -
#pragma mark File IO


JSONNode xmm::HMM::to_json() const
{
    check_training();
    JSONNode json_hmm(JSON_NODE);
    json_hmm.set_name("HMM");
    
    // Write Parent: EM Learning Model
    JSONNode json_emmodel = ProbabilisticModel::to_json();
    json_emmodel.set_name("ProbabilisticModel");
    json_hmm.push_back(json_emmodel);
    
    // Scalar Attributes
    json_hmm.push_back(JSONNode("is_hierarchical", is_hierarchical_));
    json_hmm.push_back(JSONNode("estimatemeans", estimateMeans_));
    json_hmm.push_back(JSONNode("nbstates", nbStates_));
    json_hmm.push_back(JSONNode("nbmixturecomponents", nbMixtureComponents_));
    json_hmm.push_back(JSONNode("varianceoffset_relative", varianceOffset_relative_));
    json_hmm.push_back(JSONNode("varianceoffset_absolute", varianceOffset_absolute_));
    json_hmm.push_back(JSONNode("covariance_mode", covariance_mode_));
    json_hmm.push_back(JSONNode("regression_estimator", regression_estimator_));
    json_hmm.push_back(JSONNode("transitionmode", int(transitionMode_)));
    
    // Model Parameters
    json_hmm.push_back(vector2json(prior, "prior"));
    json_hmm.push_back(vector2json(transition, "transition"));
    if (is_hierarchical_)
        json_hmm.push_back(vector2json(exitProbabilities_, "exitprobabilities"));
    
    // States
    JSONNode json_states(JSON_ARRAY);
    for (int i=0 ; i<nbStates_ ; i++)
    {
        json_states.push_back(states[i].to_json());
    }
    json_states.set_name("states");
    json_hmm.push_back(json_states);
    
    return json_hmm;
}


void xmm::HMM::from_json(JSONNode root)
{
    check_training();
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'", root.name());
        JSONNode::iterator root_it = root.begin();
        
        // Get Parent: ProbabilisticModel
        root_it = root.find("ProbabilisticModel");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NODE)
            throw JSONException("Wrong type for node 'ProbabilisticModel': was expecting 'JSON_NODE'", root_it->name());
        ProbabilisticModel::from_json(*root_it);
        
        // Get If Hierarchical
        root_it = root.find("is_hierarchical");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type for node 'is_hierarchical': was expecting 'JSON_BOOL'", root_it->name());
        if (is_hierarchical_ != root_it->as_bool()) {
            if (is_hierarchical_)
                throw JSONException("Trying to read a non-hierarchical model in a hierarchical model.", root.name());
            else
                throw JSONException("Trying to read a hierarchical model in a non-hierarchical model.", root.name());
        }
        
        // Get If estimate means
        root_it = root.find("estimatemeans");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException("Wrong type for node 'estimatemeans': was expecting 'JSON_BOOL'", root_it->name());
        estimateMeans_ = root_it->as_bool();
        
        // Get Number of states
        root_it = root.find("nbstates");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'nbstates': was expecting 'JSON_NUMBER'", root_it->name());
        nbStates_ = static_cast<int>(root_it->as_int());
        
        // Get Number of Mixture Components
        root_it = root.find("nbmixturecomponents");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'nbmixturecomponents': was expecting 'JSON_NUMBER'", root_it->name());
        nbMixtureComponents_ = static_cast<int>(root_it->as_int());
        
        // Get Covariance Offset (Relative to data variance)
        root_it = root.find("varianceoffset_relative");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'varianceoffset_relative': was expecting 'JSON_NUMBER'", root_it->name());
        varianceOffset_relative_ = root_it->as_float();
        
        // Get Covariance Offset (Minimum value)
        root_it = root.find("varianceoffset_absolute");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'varianceoffset_absolute': was expecting 'JSON_NUMBER'", root_it->name());
        varianceOffset_absolute_ = root_it->as_float();
        
        // Get Covariance mode
        root_it = root.find("covariance_mode");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_NUMBER)
                throw JSONException("Wrong type for node 'covariance_mode': was expecting 'JSON_NUMBER'", root_it->name());
            covariance_mode_ = static_cast<GaussianDistribution::COVARIANCE_MODE>(root_it->as_int());
        } else {
            covariance_mode_ = GaussianDistribution::FULL;
        }
        
        // Get Regression Estimator
        root_it = root.find("regression_estimator");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'regression_estimator': was expecting 'JSON_NUMBER'", root_it->name());
        regression_estimator_ = REGRESSION_ESTIMATOR(root_it->as_int());
        
        // Get Transition Mode
        root_it = root.find("transitionmode");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type for node 'transitionmode': was expecting 'JSON_NUMBER'", root_it->name());
        transitionMode_ = TRANSITION_MODE(root_it->as_int());
        
        // Reallocate model parameters
        allocate();
        
        // Get Prior Probabilities
        root_it = root.find("prior");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'prior': was expecting 'JSON_ARRAY'", root_it->name());
        if (transitionMode_ == ERGODIC) {
            json2vector(*root_it, prior, nbStates_);
        }
        
        // Get Transition Matrix
        root_it = root.find("transition");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'transition': was expecting 'JSON_ARRAY'", root_it->name());
        if (transitionMode_ == ERGODIC) {
            json2vector(*root_it, transition, nbStates_*nbStates_);
        } else {
            try {
                json2vector(*root_it, transition, nbStates_*2);
            } catch (std::exception& e) {
                std::cout << "warning: hazardous reading from previous file version" << std::endl;
                std::vector<double> deprec_trans(nbStates_*nbStates_);
                json2vector(*root_it, deprec_trans, nbStates_*nbStates_);
                for (unsigned int i=0; i<nbStates_-1; ++i) {
                    transition[i*2] = deprec_trans[i*nbStates_+i];
                    transition[i*2+1] = deprec_trans[i*nbStates_+i+1];
                }
                transition[(nbStates_-1)*2] = deprec_trans[nbStates_*nbStates_-1];
                transition[nbStates_*2-1] = deprec_trans[(nbStates_-1)*nbStates_];
            }
        }
        
        // Get Exit probabilities
        if (is_hierarchical_) {
            root_it = root.find("exitprobabilities");
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->type() != JSON_ARRAY)
                throw JSONException("Wrong type for node 'exitprobabilities': was expecting 'JSON_ARRAY'", root_it->name());
            json2vector(*root_it, exitProbabilities_, nbStates_);
        }
        
        // Get States
        root_it = root.find("states");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException("Wrong type for node 'states': was expecting 'JSON_ARRAY'", root_it->name());
        for (int i=0 ; i<nbStates_ ; i++) {
            states[i].from_json((*root_it)[i]);
        }
        
    } catch (JSONException &e) {
        throw JSONException(e, root.name());
    } catch (std::exception &e) {
        throw JSONException(e, root.name());
    }
    
    this->trained = true;
}

#pragma mark -
#pragma mark Exit Probabilities

void xmm::HMM::updateExitProbabilities(float *exitProbabilities)
{
    if (!is_hierarchical_)
        throw std::runtime_error("Model is Not hierarchical: method cannot be used");
    if (exitProbabilities == NULL) {
        exitProbabilities_.resize(this->nbStates_, 0.0);
        exitProbabilities_[this->nbStates_-1] = DEFAULT_EXITPROBABILITY_LAST_STATE();
    } else {
        exitProbabilities_.resize(this->nbStates_, 0.0);
        for (int i=0 ; i < this->nbStates_ ; i++)
            try {
                exitProbabilities_[i] = exitProbabilities[i];
            } catch (std::exception &e) {
                throw std::invalid_argument("Wrong format for exit probabilities");
            }
    }
}


void xmm::HMM::addExitPoint(int stateIndex, float proba)
{
    if (!is_hierarchical_)
        throw std::runtime_error("Model is Not hierarchical: method cannot be used");
    if (stateIndex >= this->nbStates_)
        throw std::out_of_range("State index out of bounds");
    exitProbabilities_[stateIndex] = proba;
}

#pragma mark > Conversion & Extraction
void xmm::HMM::make_bimodal(unsigned int dimension_input)
{
    check_training();
    if (bimodal_)
        throw std::runtime_error("The model is already bimodal");
    if (dimension_input >= dimension_)
        throw std::out_of_range("Request input dimension exceeds the current dimension");
    set_trainingSet(NULL);
    flags_ = flags_ | BIMODAL;
    bimodal_ = true;
    dimension_input_ = dimension_input;
    for (unsigned int i=0; i<nbStates_; i++) {
        states[i].make_bimodal(dimension_input);
    }
    results_predicted_output.resize(dimension_ - dimension_input_);
    results_output_variance.resize(dimension_ - dimension_input_);
}

void xmm::HMM::make_unimodal()
{
    check_training();
    if (!bimodal_)
        throw std::runtime_error("The model is already unimodal");
    set_trainingSet(NULL);
    flags_ = NONE;
    bimodal_ = false;
    dimension_input_ = 0;
    for (unsigned int i=0; i<nbStates_; i++) {
        states[i].make_unimodal();
    }
    results_predicted_output.clear();
    results_output_variance.clear();
}

xmm::HMM xmm::HMM::extract_submodel(std::vector<unsigned int>& columns) const
{
    check_training();
    if (columns.size() > dimension_)
        throw std::out_of_range("requested number of columns exceeds the dimension of the current model");
    for (unsigned int column=0; column<columns.size(); ++column) {
        if (columns[column] >= dimension_)
            throw std::out_of_range("Some column indices exceeds the dimension of the current model");
    }
    HMM target_model(*this);
    size_t new_dim = columns.size();
    target_model.set_trainingSet(NULL);
    target_model.set_trainingCallback(NULL, NULL);
    target_model.bimodal_ = false;
    target_model.dimension_ = static_cast<unsigned int>(new_dim);
    target_model.dimension_input_ = 0;
    target_model.flags_ = (this->flags_ & HIERARCHICAL);
    target_model.allocate();
    target_model.column_names_.resize(new_dim);
    for (unsigned int new_index=0; new_index<new_dim; ++new_index) {
        target_model.column_names_[new_index] = column_names_[columns[new_index]];
    }
    for (unsigned int i=0; i<nbStates_; ++i) {
        target_model.states[i] = states[i].extract_submodel(columns);
    }
    return target_model;
}

xmm::HMM xmm::HMM::extract_submodel_input() const
{
    check_training();
    if (!bimodal_)
        throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns_input(dimension_input_);
    for (unsigned int i=0; i<dimension_input_; ++i) {
        columns_input[i] = i;
    }
    return extract_submodel(columns_input);
}

xmm::HMM xmm::HMM::extract_submodel_output() const
{
    check_training();
    if (!bimodal_)
        throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns_output(dimension_ - dimension_input_);
    for (unsigned int i=dimension_input_; i<dimension_; ++i) {
        columns_output[i-dimension_input_] = i;
    }
    return extract_submodel(columns_output);
}

xmm::HMM xmm::HMM::extract_inverse_model() const
{
    check_training();
    if (!bimodal_)
        throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns(dimension_);
    for (unsigned int i=0; i<dimension_-dimension_input_; ++i) {
        columns[i] = i+dimension_input_;
    }
    for (unsigned int i=dimension_-dimension_input_, j=0; i<dimension_; ++i, ++j) {
        columns[i] = j;
    }
    HMM target_model = extract_submodel(columns);
    target_model.make_bimodal(dimension_-dimension_input_);
    return target_model;
}
