/*
 * hierarchical_hmm.cpp
 *
 * Hierarchical Hidden Markov Model for continuous recognition and regression
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

#include "hierarchical_hmm.h"
#include <algorithm>

#pragma mark -
#pragma mark Constructors
xmm::HierarchicalHMM::HierarchicalHMM(
    xmm_flags flags, TrainingSet* _globalTrainingSet,
    GaussianDistribution::COVARIANCE_MODE covariance_mode)
    : ModelGroup<HMM>(flags | HIERARCHICAL, _globalTrainingSet),
      forwardInitialized_(false) {
    set_covariance_mode(covariance_mode);
}

xmm::HierarchicalHMM::HierarchicalHMM(HierarchicalHMM const& src) {
    this->_copy(this, src);
}

xmm::HierarchicalHMM& xmm::HierarchicalHMM::operator=(
    HierarchicalHMM const& src) {
    if (this != &src) {
        if (this->globalTrainingSet)
            this->globalTrainingSet->remove_listener(this);
        _copy(this, src);
    }
    return *this;
}

void xmm::HierarchicalHMM::_copy(HierarchicalHMM* dst,
                                 HierarchicalHMM const& src) {
    ModelGroup<HMM>::_copy(dst, src);
    dst->prior = src.prior;
    dst->exitTransition = src.exitTransition;
    dst->transition = src.transition;
    dst->forwardInitialized_ = src.forwardInitialized_;
    dst->V1_ = src.V1_;
    dst->V2_ = src.V2_;
}

xmm::HierarchicalHMM::~HierarchicalHMM() {
    prior.clear();
    transition.clear();
    exitTransition.clear();
    V1_.clear();
    V2_.clear();
}

void xmm::HierarchicalHMM::clear() {
    ModelGroup<HMM>::clear();
    prior.clear();
    transition.clear();
    exitTransition.clear();
}

#pragma mark -
#pragma mark Get & Set
int xmm::HierarchicalHMM::get_nbStates() const {
    return this->referenceModel_.get_nbStates();
}

void xmm::HierarchicalHMM::set_nbStates(int nbStates_) {
    prevent_attribute_change();
    this->referenceModel_.set_nbStates(nbStates_);
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++) {
        it->second.set_nbStates(nbStates_);
    }
}

int xmm::HierarchicalHMM::get_nbMixtureComponents() const {
    return this->referenceModel_.get_nbMixtureComponents();
}

void xmm::HierarchicalHMM::set_nbMixtureComponents(int nbMixtureComponents_) {
    prevent_attribute_change();
    this->referenceModel_.set_nbMixtureComponents(nbMixtureComponents_);
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++) {
        it->second.set_nbMixtureComponents(nbMixtureComponents_);
    }
}

double xmm::HierarchicalHMM::get_varianceOffset_relative() const {
    return this->referenceModel_.get_varianceOffset_relative();
}

double xmm::HierarchicalHMM::get_varianceOffset_absolute() const {
    return this->referenceModel_.get_varianceOffset_absolute();
}

void xmm::HierarchicalHMM::set_varianceOffset(double varianceOffset_relative,
                                              double varianceOffset_absolute) {
    prevent_attribute_change();
    this->referenceModel_.set_varianceOffset(varianceOffset_relative,
                                             varianceOffset_absolute);
    for (model_iterator it = this->models.begin(); it != this->models.end();
         ++it) {
        it->second.set_varianceOffset(varianceOffset_relative,
                                      varianceOffset_absolute);
    }
}

xmm::GaussianDistribution::COVARIANCE_MODE
xmm::HierarchicalHMM::get_covariance_mode() const {
    return this->referenceModel_.get_covariance_mode();
}

void xmm::HierarchicalHMM::set_covariance_mode(
    GaussianDistribution::COVARIANCE_MODE covariance_mode) {
    prevent_attribute_change();
    if (covariance_mode == get_covariance_mode()) return;
    try {
        this->referenceModel_.set_covariance_mode(covariance_mode);
    } catch (std::exception& e) {
        if (strncmp(e.what(), "Non-invertible matrix", 21) != 0)
            throw std::runtime_error(e.what());
    }
    for (model_iterator it = this->models.begin(); it != this->models.end();
         ++it) {
        it->second.set_covariance_mode(covariance_mode);
    }
}

xmm::HMM::REGRESSION_ESTIMATOR xmm::HierarchicalHMM::get_regression_estimator()
    const {
    return this->referenceModel_.get_regression_estimator();
}

void xmm::HierarchicalHMM::set_regression_estimator(
    xmm::HMM::REGRESSION_ESTIMATOR regression_estimator) {
    prevent_attribute_change();
    this->referenceModel_.set_regression_estimator(regression_estimator);
    for (model_iterator it = this->models.begin(); it != this->models.end();
         ++it) {
        it->second.set_regression_estimator(regression_estimator);
    }
}

bool xmm::HierarchicalHMM::get_estimateMeans() const {
    return this->referenceModel_.estimateMeans_;
}

void xmm::HierarchicalHMM::set_estimateMeans(bool _estimateMeans) {
    prevent_attribute_change();
    this->referenceModel_.estimateMeans_ = _estimateMeans;
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++)
        it->second.estimateMeans_ = _estimateMeans;
}

std::string xmm::HierarchicalHMM::get_transitionMode() const {
    return this->referenceModel_.get_transitionMode();
}

void xmm::HierarchicalHMM::set_transitionMode(std::string transMode_str) {
    prevent_attribute_change();
    this->referenceModel_.set_transitionMode(transMode_str);
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++) {
        it->second.set_transitionMode(transMode_str);
    }
}

void xmm::HierarchicalHMM::addExitPoint(int state, float proba) {
    prevent_attribute_change();
    this->referenceModel_.addExitPoint(state, proba);
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++) {
        it->second.addExitPoint(state, proba);
    }
}

void xmm::HierarchicalHMM::get_prior(std::vector<double>& prior_) const {
    prior_ = this->prior;
}

void xmm::HierarchicalHMM::set_prior(std::vector<double> const& prior_) {
    try {
        for (int i = 0; i < size(); i++) {
            this->prior[i] = std::max(prior_[i], 0.0);
        }
        this->normalizeTransitions();
    } catch (std::exception& e) {
        throw std::invalid_argument("Wrong format for prior");
    }
}

void xmm::HierarchicalHMM::get_transition(std::vector<double>& trans) const {
    unsigned int nbPrimitives = this->size();
    trans.resize(nbPrimitives * nbPrimitives);
    for (int i = 0; i < nbPrimitives; i++) {
        for (int j = 0; j < nbPrimitives; j++) {
            trans[i * nbPrimitives + j] = this->transition[i][j];
        }
    }
}

void xmm::HierarchicalHMM::set_transition(std::vector<double> const& trans) {
    try {
        unsigned int nbPrimitives = this->size();
        for (int i = 0; i < nbPrimitives; i++) {
            for (int j = 0; j < nbPrimitives; j++) {
                this->transition[i][j] =
                    std::max(trans[i * nbPrimitives + j], 0.0);
            }
        }
        this->normalizeTransitions();
    } catch (std::exception& e) {
        throw std::invalid_argument("Wrong format for transition");
    }
}

void xmm::HierarchicalHMM::get_exitTransition(
    std::vector<double>& exittrans) const {
    exittrans = this->exitTransition;
}

void xmm::HierarchicalHMM::set_exitTransition(
    std::vector<double> const& exittrans) {
    try {
        for (int i = 0; i < size(); i++) {
            this->exitTransition[i] = std::max(exittrans[i], 0.0);
        }
        this->normalizeTransitions();
    } catch (std::exception& e) {
        throw std::invalid_argument("Wrong format for prior");
    }
}

void xmm::HierarchicalHMM::normalizeTransitions() {
    unsigned int nbPrimitives = this->size();
    double sumPrior(0.0);
    for (int i = 0; i < nbPrimitives; i++) {
        sumPrior += prior[i];

        double sumTrans(0.0);
        for (int j = 0; j < nbPrimitives; j++) sumTrans += transition[i][j];
        if (sumTrans > 0.0)
            for (int j = 0; j < nbPrimitives; j++) transition[i][j] /= sumTrans;
    }
    for (int i = 0; i < nbPrimitives; i++) prior[i] /= sumPrior;
}

void xmm::HierarchicalHMM::setOneTransition(Label srcSegmentLabel,
                                            Label dstSegmentLabel,
                                            double proba) {
    int src_index(-1);
    int dst_index(-1);
    int i(0);
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++, i++) {
        if (it->first == srcSegmentLabel) src_index = i;
        if (it->first == dstSegmentLabel) dst_index = i;
        if (src_index >= 0 && dst_index >= 0) break;
    }
    if (src_index >= 0 && dst_index >= 0) {
        transition[src_index][dst_index] = std::max(std::min(proba, 1.), 0.);
        normalizeTransitions();
    }
}

#pragma mark -
#pragma mark High level parameters: update and estimation

void xmm::HierarchicalHMM::updateTransitionParameters() {
    updatePrior_ergodic();
    updateTransition_ergodic();
    updateExitProbabilities();
}

void xmm::HierarchicalHMM::updatePrior_ergodic() {
    int nbPrimitives = this->size();
    prior.assign(nbPrimitives, 1. / double(nbPrimitives));
}

void xmm::HierarchicalHMM::updateTransition_ergodic() {
    int nbPrimitives = this->size();
    exitTransition.assign(nbPrimitives, DEFAULT_EXITTRANSITION());
    transition.resize(nbPrimitives);
    for (int i = 0; i < nbPrimitives; i++)
        transition[i].assign(nbPrimitives, 1. / double(nbPrimitives));
}

void xmm::HierarchicalHMM::updateExitProbabilities() {
    for (model_iterator it = this->models.begin(); it != this->models.end();
         ++it) {
        it->second.updateExitProbabilities();
    }
}

void xmm::HierarchicalHMM::updateTrainingSet(Label const& label) {
    ModelGroup<HMM>::updateTrainingSet(label);
    updateTransitionParameters();
}

#pragma mark -
#pragma mark Forward Algorithm
void xmm::HierarchicalHMM::forward_init(std::vector<float> const& observation) {
    check_training();
    double norm_const(0.0);

    int model_index(0);
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++) {
        int N = it->second.get_nbStates();

        for (int i = 0; i < 3; i++) {
            it->second.alpha_h[i].assign(N, 0.0);
        }

        // Compute Emission probability and initialize on the first state of the
        // primitive
        if (it->second.transitionMode_ == HMM::ERGODIC) {
            for (int i = 0; i < it->second.nbStates_; i++) {
                if (bimodal_) {
                    it->second.alpha_h[0][i] =
                        it->second.prior[i] *
                        it->second.states[i].obsProb_input(&observation[0]);
                } else {
                    it->second.alpha_h[0][i] =
                        it->second.prior[i] *
                        it->second.states[i].obsProb(&observation[0]);
                }
                it->second.results_instant_likelihood +=
                    it->second.alpha_h[0][i];
            }
        } else {
            it->second.alpha_h[0][0] = this->prior[model_index];
            if (bimodal_) {
                it->second.alpha_h[0][0] *=
                    it->second.states[0].obsProb_input(&observation[0]);
            } else {
                it->second.alpha_h[0][0] *=
                    it->second.states[0].obsProb(&observation[0]);
            }
            it->second.results_instant_likelihood = it->second.alpha_h[0][0];
        }
        it->second.updateLikelihoodBuffer(
            it->second.results_instant_likelihood);
        norm_const += it->second.results_instant_likelihood;
        model_index++;
    }

    // Normalize Alpha variables
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++) {
        int N = it->second.get_nbStates();
        for (int e = 0; e < 3; e++)
            for (int k = 0; k < N; k++) it->second.alpha_h[e][k] /= norm_const;
    }

    forwardInitialized_ = true;
}

void xmm::HierarchicalHMM::forward_update(
    std::vector<float> const& observation) {
    check_training();
    double norm_const(0.0);

    // Frontier Algorithm: variables
    double tmp(0);
    std::vector<double>
        front;  // frontier variable : intermediate computation variable

    // Intermediate variables: compute the sum of probabilities of making a
    // transition to a new primitive
    likelihoodAlpha(1, V1_);
    likelihoodAlpha(2, V2_);

    // FORWARD UPDATE
    // --------------------------------------
    int dst_model_index(0);
    for (model_iterator dstit = this->models.begin();
         dstit != this->models.end(); dstit++) {
        int N = dstit->second.get_nbStates();

        // 1) COMPUTE FRONTIER VARIABLE
        //    --------------------------------------
        front.assign(N, 0.0);

        if (dstit->second.transitionMode_ == HMM::ERGODIC) {
            for (int k = 0; k < N; ++k) {
                for (unsigned int j = 0; j < N; ++j) {
                    front[k] += dstit->second.transition[j * N + k] /
                                (1 - dstit->second.exitProbabilities_[j]) *
                                dstit->second.alpha_h[0][j];
                }

                int src_model_index(0);
                for (model_iterator srcit = this->models.begin();
                     srcit != this->models.end(); srcit++, src_model_index++) {
                    front[k] +=
                        dstit->second.prior[k] *
                        (V1_[src_model_index] *
                             this->transition[src_model_index]
                                             [dst_model_index] +
                         this->prior[dst_model_index] * V2_[src_model_index]);
                }
            }
        } else {
            // k=0: first state of the primitive
            front[0] =
                dstit->second.transition[0] * dstit->second.alpha_h[0][0];

            int src_model_index(0);
            for (model_iterator srcit = this->models.begin();
                 srcit != this->models.end(); srcit++, src_model_index++) {
                front[0] +=
                    V1_[src_model_index] *
                        this->transition[src_model_index][dst_model_index] +
                    this->prior[dst_model_index] * V2_[src_model_index];
            }

            // k>0: rest of the primitive
            for (int k = 1; k < N; ++k) {
                front[k] += dstit->second.transition[k * 2] /
                            (1 - dstit->second.exitProbabilities_[k]) *
                            dstit->second.alpha_h[0][k];
                front[k] += dstit->second.transition[(k - 1) * 2 + 1] /
                            (1 - dstit->second.exitProbabilities_[k - 1]) *
                            dstit->second.alpha_h[0][k - 1];
            }

            for (int i = 0; i < 3; i++) {
                for (int k = 0; k < N; k++) {
                    dstit->second.alpha_h[i][k] = 0.0;
                }
            }
        }

        // 2) UPDATE FORWARD VARIABLE
        //    --------------------------------------

        dstit->second.results_exit_likelihood = 0.0;
        dstit->second.results_instant_likelihood = 0.0;

        // end of the primitive: handle exit states
        for (int k = 0; k < N; ++k) {
            if (bimodal_)
                tmp = dstit->second.states[k].obsProb_input(&observation[0]) *
                      front[k];
            else
                tmp =
                    dstit->second.states[k].obsProb(&observation[0]) * front[k];

            dstit->second.alpha_h[2][k] =
                this->exitTransition[dst_model_index] *
                dstit->second.exitProbabilities_[k] * tmp;
            dstit->second.alpha_h[1][k] =
                (1 - this->exitTransition[dst_model_index]) *
                dstit->second.exitProbabilities_[k] * tmp;
            dstit->second.alpha_h[0][k] =
                (1 - dstit->second.exitProbabilities_[k]) * tmp;

            dstit->second.results_exit_likelihood +=
                dstit->second.alpha_h[1][k] + dstit->second.alpha_h[2][k];
            dstit->second.results_instant_likelihood +=
                dstit->second.alpha_h[0][k] + dstit->second.alpha_h[1][k] +
                dstit->second.alpha_h[2][k];

            norm_const += tmp;
        }

        dstit->second.updateLikelihoodBuffer(
            dstit->second.results_instant_likelihood);
        dstit->second.results_exit_ratio =
            dstit->second.results_exit_likelihood /
            dstit->second.results_instant_likelihood;

        dst_model_index++;
    }

    // Normalize Alpha variables
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++) {
        int N = it->second.get_nbStates();
        for (int e = 0; e < 3; e++)
            for (int k = 0; k < N; k++) it->second.alpha_h[e][k] /= norm_const;
    }
}

void xmm::HierarchicalHMM::likelihoodAlpha(
    int exitNum, std::vector<double>& likelihoodVector) const {
    if (exitNum < 0) {  // Likelihood over all exit states
        int l(0);
        for (const_model_iterator it = this->models.begin();
             it != this->models.end(); it++) {
            likelihoodVector[l] = 0.0;
            for (int exit = 0; exit < 3; ++exit) {
                for (int k = 0; k < it->second.get_nbStates(); k++) {
                    likelihoodVector[l] += it->second.alpha_h[exit][k];
                }
            }
            l++;
        }

    } else {  // Likelihood for exit state "exitNum"
        int l(0);
        for (const_model_iterator it = this->models.begin();
             it != this->models.end(); it++) {
            likelihoodVector[l] = 0.0;
            for (int k = 0; k < it->second.get_nbStates(); k++) {
                likelihoodVector[l] += it->second.alpha_h[exitNum][k];
            }
            l++;
        }
    }
}

void xmm::HierarchicalHMM::remove(Label const& label) {
    ModelGroup<HMM>::remove(label);
    updateTransitionParameters();
}

#pragma mark -
#pragma mark Playing
void xmm::HierarchicalHMM::performance_init() {
    check_training();
    ModelGroup<HMM>::performance_init();
    V1_.resize(this->size());
    V2_.resize(this->size());
    forwardInitialized_ = false;
}

void xmm::HierarchicalHMM::performance_update(
    std::vector<float> const& observation) {
    check_training();
    if (forwardInitialized_) {
        this->forward_update(observation);
    } else {
        this->forward_init(observation);
    }

    update_likelihood_results();

    // Compute time progression
    for (model_iterator it = this->models.begin(); it != this->models.end();
         it++) {
        it->second.updateAlphaWindow();
        it->second.updateTimeProgression();
    }

    if (bimodal_) {
        unsigned int dimension = this->referenceModel_.dimension();
        unsigned int dimension_input = this->referenceModel_.dimension_input();
        unsigned int dimension_output = dimension - dimension_input;

        for (model_iterator it = this->models.begin(); it != this->models.end();
             ++it) {
            it->second.regression(observation,
                                  it->second.results_predicted_output);
        }

        if (this->performanceMode_ == this->LIKELIEST) {
            copy(this->models[results_likeliest]
                     .results_predicted_output.begin(),
                 this->models[results_likeliest].results_predicted_output.end(),
                 results_predicted_output.begin());
            copy(
                this->models[results_likeliest].results_output_variance.begin(),
                this->models[results_likeliest].results_output_variance.end(),
                results_output_variance.begin());
        } else {
            results_predicted_output.assign(dimension_output, 0.0);
            results_output_variance.assign(dimension_output, 0.0);

            int i(0);
            for (model_iterator it = this->models.begin();
                 it != this->models.end(); it++) {
                for (int d = 0; d < dimension_output; d++) {
                    results_predicted_output[d] +=
                        results_normalized_likelihoods[i] *
                        it->second.results_predicted_output[d];
                    results_output_variance[d] +=
                        results_normalized_likelihoods[i] *
                        it->second.results_output_variance[d];
                }
                i++;
            }
        }
    }
}

#pragma mark -
#pragma mark File IO
JSONNode xmm::HierarchicalHMM::to_json() const {
    check_training();
    JSONNode json_hhmm(JSON_NODE);
    json_hhmm.set_name("HierarchicalHMM");
    json_hhmm.push_back(JSONNode("bimodal", bimodal_));
    json_hhmm.push_back(JSONNode("dimension", dimension()));
    if (bimodal_)
        json_hhmm.push_back(JSONNode("dimension_input", dimension_input()));
    JSONNode json_stopcriterion(JSON_NODE);
    json_stopcriterion.set_name("EMStopCriterion");
    json_stopcriterion.push_back(JSONNode("minsteps", get_EM_minSteps()));
    json_stopcriterion.push_back(JSONNode("maxsteps", get_EM_maxSteps()));
    json_stopcriterion.push_back(
        JSONNode("percentchg", get_EM_percentChange()));
    json_hhmm.push_back(json_stopcriterion);
    json_hhmm.push_back(JSONNode("likelihoodwindow", get_likelihoodwindow()));
    json_hhmm.push_back(JSONNode("estimatemeans", get_estimateMeans()));
    json_hhmm.push_back(JSONNode("size", models.size()));
    json_hhmm.push_back(JSONNode("performancemode", int(performanceMode_)));
    json_hhmm.push_back(JSONNode("nbstates", get_nbStates()));
    json_hhmm.push_back(
        JSONNode("nbmixturecomponents", get_nbMixtureComponents()));
    json_hhmm.push_back(
        JSONNode("varianceoffset_relative", get_varianceOffset_relative()));
    json_hhmm.push_back(
        JSONNode("varianceoffset_absolute", get_varianceOffset_absolute()));
    json_hhmm.push_back(JSONNode("covariance_mode", get_covariance_mode()));
    json_hhmm.push_back(
        JSONNode("regression_estimator", get_regression_estimator()));

    // Add Models
    JSONNode json_models(JSON_ARRAY);
    for (const_model_iterator it = models.begin(); it != models.end(); ++it) {
        JSONNode json_model(JSON_NODE);
        json_model.push_back(it->first.to_json());
        json_model.push_back(it->second.to_json());
        json_models.push_back(json_model);
    }
    json_models.set_name("models");
    json_hhmm.push_back(json_models);

    json_hhmm.push_back(vector2json(prior, "prior"));
    json_hhmm.push_back(vector2json(exitTransition, "exit"));

    int nbPrimitives = this->size();
    std::vector<double> trans(nbPrimitives * nbPrimitives);
    for (int i = 0; i < nbPrimitives; i++)
        for (int j = 0; j < nbPrimitives; j++)
            trans[i * nbPrimitives + j] = transition[i][j];
    json_hhmm.push_back(vector2json(trans, "transition"));

    return json_hhmm;
}

void xmm::HierarchicalHMM::from_json(JSONNode root) {
    check_training();
    try {
        if (root.type() != JSON_NODE)
            throw JSONException("Wrong type: was expecting 'JSON_NODE'",
                                root.name());
        JSONNode::const_iterator root_it = root.begin();

        // Get Number of modalities
        root_it = root.find("bimodal");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_BOOL)
            throw JSONException(
                "Wrong type for node 'bimodal': was expecting 'JSON_BOOL'",
                root_it->name());
        if (bimodal_ != root_it->as_bool()) {
            if (bimodal_)
                throw JSONException(
                    "Trying to read an unimodal model in a bimodal model.",
                    root.name());
            else
                throw JSONException(
                    "Trying to read a bimodal model in an unimodal model.",
                    root.name());
        }

        // Get Dimension
        root_it = root.find("dimension");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException(
                "Wrong type for node 'dimension': was expecting "
                "'JSON_NUMBER'",
                root_it->name());
        this->referenceModel_.dimension_ =
            static_cast<unsigned int>(root_it->as_int());

        // Get Input Dimension if bimodal
        if (bimodal_) {
            root_it = root.find("dimension_input");
            if (root_it == root.end())
                throw JSONException("JSON Node is incomplete", root_it->name());
            if (root_it->type() != JSON_NUMBER)
                throw JSONException(
                    "Wrong type for node 'dimension_input': was expecting "
                    "'JSON_NUMBER'",
                    root_it->name());
            this->referenceModel_.dimension_input_ =
                static_cast<unsigned int>(root_it->as_int());
        }

        // Get EM Algorithm stop criterion
        root_it = root.find("EMStopCriterion");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NODE)
            throw JSONException(
                "Wrong type for node 'EMStopCriterion': was expecting "
                "'JSON_NODE'",
                root_it->name());
        JSONNode json_stopcriterion = *root_it;
        JSONNode::const_iterator crit_it = json_stopcriterion.begin();
        if (crit_it == json_stopcriterion.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "minsteps")
            throw JSONException("Wrong name: was expecting 'minsteps'",
                                crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'",
                                crit_it->name());
        set_EM_minSteps(static_cast<unsigned int>(crit_it->as_int()));
        crit_it++;

        if (crit_it == json_stopcriterion.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "maxsteps")
            throw JSONException("Wrong name: was expecting 'maxsteps'",
                                crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'",
                                crit_it->name());
        set_EM_maxSteps(static_cast<unsigned int>(crit_it->as_int()));
        crit_it++;

        if (crit_it == json_stopcriterion.end())
            throw JSONException("JSON Node is incomplete", crit_it->name());
        if (crit_it->name() != "percentchg")
            throw JSONException("Wrong name: was expecting 'percentchg'",
                                crit_it->name());
        if (crit_it->type() != JSON_NUMBER)
            throw JSONException("Wrong type: was expecting 'JSON_NUMBER'",
                                crit_it->name());
        set_EM_percentChange(crit_it->as_float());

        // Get likelihood window size
        root_it = root.find("likelihoodwindow");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_NUMBER)
                throw JSONException(
                    "Wrong type for node 'likelihoodwindow': was expecting "
                    "'JSON_NUMBER'",
                    root_it->name());
            this->set_likelihoodwindow(
                static_cast<unsigned int>(root_it->as_int()));
        }

        // Get If Estimate Means
        root_it = root.find("estimatemeans");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_BOOL)
                throw JSONException(
                    "Wrong type for node 'estimatemeans': was expecting "
                    "'JSON_BOOL'",
                    root_it->name());
            this->set_estimateMeans(root_it->as_bool());
        }

        // Get Size: Number of Models
        root_it = root.find("size");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException(
                "Wrong type for node 'size': was expecting 'JSON_NUMBER'",
                root_it->name());
        unsigned int numModels = static_cast<unsigned int>(root_it->as_int());

        // Get Play Mode
        root_it = root.find("performancemode");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException(
                "Wrong type for node 'performancemode': was expecting "
                "'JSON_NUMBER'",
                root_it->name());
        performanceMode_ = (root_it->as_int() > 0) ? MIXTURE : LIKELIEST;

        // Get Number of States
        root_it = root.find("nbstates");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException(
                "Wrong type for node 'nbstates': was expecting "
                "'JSON_NUMBER'",
                root_it->name());
        set_nbStates(static_cast<int>(root_it->as_int()));

        // Get Number of Mixture Components
        root_it = root.find("nbmixturecomponents");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException(
                "Wrong type for node 'nbmixturecomponents': was expecting "
                "'JSON_NUMBER'",
                root_it->name());
        set_nbMixtureComponents(static_cast<int>(root_it->as_int()));

        // Get Covariance Offset (Relative to data variance)
        root_it = root.find("varianceoffset_relative");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException(
                "Wrong type for node 'varianceoffset_relative': was "
                "expecting "
                "'JSON_NUMBER'",
                root_it->name());
        double relvar = root_it->as_float();

        // Get Covariance Offset (Minimum value)
        root_it = root.find("varianceoffset_absolute");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException(
                "Wrong type for node 'varianceoffset_absolute': was "
                "expecting "
                "'JSON_NUMBER'",
                root_it->name());
        set_varianceOffset(relvar, root_it->as_float());

        // Get Covariance mode
        root_it = root.find("covariance_mode");
        if (root_it != root.end()) {
            if (root_it->type() != JSON_NUMBER)
                throw JSONException(
                    "Wrong type for node 'covariance_mode': was expecting "
                    "'JSON_NUMBER'",
                    root_it->name());
            set_covariance_mode(
                static_cast<GaussianDistribution::COVARIANCE_MODE>(
                    root_it->as_int()));
        } else {
            set_covariance_mode(GaussianDistribution::FULL);
        }

        // Get Regression Estimator
        root_it = root.find("regression_estimator");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_NUMBER)
            throw JSONException(
                "Wrong type for node 'regression_estimator': was expecting "
                "'JSON_NUMBER'",
                root_it->name());
        set_regression_estimator(
            xmm::HMM::REGRESSION_ESTIMATOR(root_it->as_int()));

        // Get Models
        models.clear();
        root_it = root.find("models");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException(
                "Wrong type for node 'models': was expecting 'JSON_ARRAY'",
                root_it->name());
        for (unsigned int i = 0; i < numModels; i++) {
            // Get Label
            JSONNode::const_iterator array_it = (*root_it)[i].begin();
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete",
                                    array_it->name());
            if (array_it->name() != "label")
                throw JSONException("Wrong name: was expecting 'label'",
                                    array_it->name());
            if (array_it->type() != JSON_NODE)
                throw JSONException("Wrong type: was expecting 'JSON_NODE'",
                                    array_it->name());
            Label l;
            l.from_json(*array_it);
            ++array_it;

            // Get Phrase Content
            if (array_it == root.end())
                throw JSONException("JSON Node is incomplete",
                                    array_it->name());
            if (array_it->name() != "HMM")
                throw JSONException("Wrong name: was expecting 'HMM'",
                                    array_it->name());
            if (array_it->type() != JSON_NODE)
                throw JSONException("Wrong type: was expecting 'JSON_NODE'",
                                    array_it->name());
            models[l] = this->referenceModel_;
            models[l].from_json(*array_it);
        }
        if (numModels != models.size())
            throw JSONException("Number of models does not match", root.name());

        // Get High-level Prior
        root_it = root.find("prior");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException(
                "Wrong type for node 'prior': was expecting "
                "'JSON_ARRAY'",
                root_it->name());
        json2vector(*root_it, prior, numModels);

        // Get High-level Exit Probabilities
        root_it = root.find("exit");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException(
                "Wrong type for node 'exit': was expecting "
                "'JSON_ARRAY'",
                root_it->name());
        json2vector(*root_it, exitTransition, numModels);

        // Get High-level Transition Matrix
        root_it = root.find("transition");
        if (root_it == root.end())
            throw JSONException("JSON Node is incomplete", root_it->name());
        if (root_it->type() != JSON_ARRAY)
            throw JSONException(
                "Wrong type for node 'transition': was expecting "
                "'JSON_ARRAY'",
                root_it->name());
        std::vector<double> trans(numModels * numModels);
        json2vector(*root_it, trans, numModels * numModels);
        transition.resize(numModels);
        for (int i = 0; i < numModels; i++)
            for (int j = 0; j < numModels; j++)
                transition[i][j] = trans[i * numModels + j];

    } catch (JSONException& e) {
        throw JSONException(e, root.name());
    } catch (std::exception& e) {
        throw JSONException(e, root.name());
    }
}

#pragma mark > Conversion & Extraction
void xmm::HierarchicalHMM::make_bimodal(unsigned int dimension_input) {
    check_training();
    if (bimodal_) throw std::runtime_error("The model is already bimodal");
    if (dimension_input >= dimension())
        throw std::out_of_range(
            "Request input dimension exceeds the current dimension");

    try {
        this->referenceModel_.make_bimodal(dimension_input);
    } catch (std::exception const& e) {
    }
    bimodal_ = true;
    for (model_iterator it = this->models.begin(); it != this->models.end();
         ++it) {
        it->second.make_bimodal(dimension_input);
    }
    set_trainingSet(NULL);
    results_predicted_output.resize(dimension() - this->dimension_input());
    results_output_variance.resize(dimension() - this->dimension_input());
}

void xmm::HierarchicalHMM::make_unimodal() {
    check_training();
    if (!bimodal_) throw std::runtime_error("The model is already unimodal");
    this->referenceModel_.make_unimodal();
    for (model_iterator it = this->models.begin(); it != this->models.end();
         ++it) {
        it->second.make_unimodal();
    }
    set_trainingSet(NULL);
    results_predicted_output.clear();
    results_output_variance.clear();
    bimodal_ = false;
}

xmm::HierarchicalHMM xmm::HierarchicalHMM::extract_submodel(
    std::vector<unsigned int>& columns) const {
    check_training();
    if (columns.size() > this->dimension())
        throw std::out_of_range(
            "requested number of columns exceeds the dimension of the "
            "current "
            "model");
    for (unsigned int column = 0; column < columns.size(); ++column) {
        if (columns[column] >= this->dimension())
            throw std::out_of_range(
                "Some column indices exceeds the dimension of the current "
                "model");
    }
    HierarchicalHMM target_model(*this);
    target_model.set_trainingSet(NULL);
    target_model.bimodal_ = false;
    target_model.referenceModel_ =
        this->referenceModel_.extract_submodel(columns);
    for (model_iterator it = target_model.models.begin();
         it != target_model.models.end(); ++it) {
        it->second = this->models.at(it->first).extract_submodel(columns);
    }
    return target_model;
}

xmm::HierarchicalHMM xmm::HierarchicalHMM::extract_submodel_input() const {
    check_training();
    if (!bimodal_) throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns_input(dimension_input());
    for (unsigned int i = 0; i < dimension_input(); ++i) {
        columns_input[i] = i;
    }
    return extract_submodel(columns_input);
}

xmm::HierarchicalHMM xmm::HierarchicalHMM::extract_submodel_output() const {
    check_training();
    if (!bimodal_) throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns_output(dimension() - dimension_input());
    for (unsigned int i = dimension_input(); i < dimension(); ++i) {
        columns_output[i - dimension_input()] = i;
    }
    return extract_submodel(columns_output);
}

xmm::HierarchicalHMM xmm::HierarchicalHMM::extract_inverse_model() const {
    check_training();
    if (!bimodal_) throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns(dimension());
    for (unsigned int i = 0; i < dimension() - dimension_input(); ++i) {
        columns[i] = i + dimension_input();
    }
    for (unsigned int i = dimension() - dimension_input(), j = 0;
         i < dimension(); ++i, ++j) {
        columns[i] = j;
    }
    HierarchicalHMM target_model = extract_submodel(columns);
    target_model.make_bimodal(dimension() - dimension_input());
    return target_model;
}
