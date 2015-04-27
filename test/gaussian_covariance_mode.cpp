/*
 * gaussian_covariance_mode.cpp
 *
 * Test suite for Diagonal vs Full covariance Gaussian Distributions
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

#include "catch.hpp"
#include "catch_utilities.h"
#define XMM_TESTING
#include "xmm.h"
#include <ctime>

TEST_CASE( "Diagonal covariance (unimodal)", "[GaussianDistribution]" ) {
    GaussianDistribution a(NONE,
                           3,
                           0,
                           0.0034,
                           0.0123);
    a.mean[0] = 0.2;
    a.mean[1] = 0.3;
    a.mean[2] = 0.1;
    a.covariance[0] = 1.3;
    a.covariance[1] = 0.0;
    a.covariance[2] = 0.2;
    a.covariance[3] = 0.0;
    a.covariance[4] = 1.4;
    a.covariance[5] = 0.7;
    a.covariance[6] = 0.2;
    a.covariance[7] = 0.7;
    a.covariance[8] = 1.5;
    a.updateInverseCovariance();
    float *observation = new float[3];
    observation[0] = 0.7;
    observation[1] = 0.;
    observation[2] = -0.3;
    GaussianDistribution b(a);
    CHECK_NOTHROW(b.set_covariance_mode(GaussianDistribution::DIAGONAL));
    CHECK(a.mean == b.mean);
    CHECK(a.covariance[0] == b.covariance[0]);
    CHECK(a.covariance[4] == b.covariance[1]);
    CHECK(a.covariance[8] == b.covariance[2]);
    double likelihood_b = b.likelihood(observation);
    GaussianDistribution c(b);
    CHECK_NOTHROW(c.set_covariance_mode(GaussianDistribution::FULL));
    CHECK(a.mean == c.mean);
    CHECK_FALSE(a.covariance == c.covariance);
    CHECK(c.covariance[0] == b.covariance[0]);
    CHECK(c.covariance[4] == b.covariance[1]);
    CHECK(c.covariance[8] == b.covariance[2]);
    double likelihood_c = c.likelihood(observation);
    CHECK(likelihood_b == likelihood_c);
    delete[] observation;
}


TEST_CASE( "Diagonal covariance (bimodal)", "[GaussianDistribution]" ) {
    GaussianDistribution a(BIMODAL,
                           3,
                           2,
                           0.0034,
                           0.0123);
    a.mean[0] = 0.2;
    a.mean[1] = 0.3;
    a.mean[2] = 0.1;
    a.covariance[0] = 1.3;
    a.covariance[1] = 0.8;
    a.covariance[2] = 0.2;
    a.covariance[3] = 0.8;
    a.covariance[4] = 1.4;
    a.covariance[5] = 0.7;
    a.covariance[6] = 0.2;
    a.covariance[7] = 0.7;
    a.covariance[8] = 1.5;
    a.updateInverseCovariance();
    float *observation = new float[3];
    observation[0] = 0.7;
    observation[1] = 0.;
    observation[2] = -0.3;
    GaussianDistribution b(a);
    CHECK_NOTHROW(b.set_covariance_mode(GaussianDistribution::DIAGONAL));
    CHECK(a.mean == b.mean);
    CHECK(a.covariance[0] == b.covariance[0]);
    CHECK(a.covariance[4] == b.covariance[1]);
    CHECK(a.covariance[8] == b.covariance[2]);
    GaussianDistribution c(b);
    CHECK_NOTHROW(c.set_covariance_mode(GaussianDistribution::FULL));
    CHECK(a.mean == c.mean);
    CHECK_FALSE(a.covariance == c.covariance);
    CHECK(c.covariance[0] == b.covariance[0]);
    CHECK(c.covariance[4] == b.covariance[1]);
    CHECK(c.covariance[8] == b.covariance[2]);
    CHECK(b.likelihood_input(observation) == c.likelihood_input(observation));
    CHECK_FALSE(b.likelihood_input(observation) == a.likelihood_input(observation));
    delete[] observation;
}

TEST_CASE( "GMM with Diagonal covariance (unimodal)", "[GMM]" ) {
    TrainingSet ts(NONE, 3);
    vector<float> observation(3);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.recordPhrase(0, observation);
    }
    GMM a(NONE, &ts);
    a.set_nbMixtureComponents(3);
    a.train();
    CHECK_NOTHROW(a.set_covariance_mode(GaussianDistribution::DIAGONAL));
    a.performance_init();
    vector<double> log_likelihood(100, 0.0);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        a.performance_update(observation);
        log_likelihood[i] = a.results_log_likelihood;
    }
    CHECK_NOTHROW(a.set_covariance_mode(GaussianDistribution::FULL));
    a.performance_init();
    vector<double> log_likelihood2(100, 0.0);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        a.performance_update(observation);
        log_likelihood2[i] = a.results_log_likelihood;
    }
    CHECK_VECTOR_APPROX(log_likelihood, log_likelihood2);
    CHECK_NOTHROW(a.set_covariance_mode(GaussianDistribution::DIAGONAL));
    JSONNode a_json = a.to_json();
    GMM b;
    b.from_json(a_json);
}

TEST_CASE( "GMMGroup with Diagonal covariance (unimodal)", "[GMMGroup]" ) {
    TrainingSet ts(NONE, 3);
    vector<float> observation(3);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.recordPhrase(0, observation);
        ts.recordPhrase(1, observation);
    }
    Label label_a(static_cast<string>("a"));
    Label label_b(static_cast<string>("b"));
    ts.setPhraseLabel(0, label_a);
    ts.setPhraseLabel(1, label_b);
    GMMGroup a(NONE, &ts);
    a.set_nbMixtureComponents(3);
    a.train();
    CHECK_NOTHROW(a.set_covariance_mode(GaussianDistribution::DIAGONAL));
    a.performance_init();
    vector<double> log_likelihood(100, 0.0);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        a.performance_update(observation);
        log_likelihood[i] = a.results_log_likelihoods[0];
    }
    CHECK_NOTHROW(a.set_covariance_mode(GaussianDistribution::FULL));
    a.performance_init();
    vector<double> log_likelihood2(100, 0.0);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        a.performance_update(observation);
        log_likelihood2[i] = a.results_log_likelihoods[0];
    }
    CHECK_VECTOR_APPROX(log_likelihood, log_likelihood2);
}

TEST_CASE( "HMM with Diagonal covariance (unimodal)", "[HMM]" ) {
    TrainingSet ts(NONE, 3);
    vector<float> observation(3);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.recordPhrase(0, observation);
    }
    HMM a(NONE, &ts);
    a.set_nbStates(3);
    a.train();
    a.set_covariance_mode(GaussianDistribution::DIAGONAL);
    a.performance_init();
    vector<double> log_likelihood(100, 0.0);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        a.performance_update(observation);
        log_likelihood[i] = a.results_log_likelihood;
    }
    a.set_covariance_mode(GaussianDistribution::FULL);
    a.performance_init();
    vector<double> log_likelihood2(100, 0.0);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        a.performance_update(observation);
        log_likelihood2[i] = a.results_log_likelihood;
    }
    CHECK_VECTOR_APPROX(log_likelihood, log_likelihood2);
    CHECK_NOTHROW(a.set_covariance_mode(GaussianDistribution::DIAGONAL));
    JSONNode a_json = a.to_json();
    HMM b;
    b.from_json(a_json);
}

TEST_CASE( "HierarchicalHMM with Diagonal covariance (unimodal)", "[HierarchicalHMM]" ) {
    TrainingSet ts(NONE, 3);
    vector<float> observation(3);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.recordPhrase(0, observation);
        ts.recordPhrase(1, observation);
    }
    Label label_a(static_cast<string>("a"));
    Label label_b(static_cast<string>("b"));
    ts.setPhraseLabel(0, label_a);
    ts.setPhraseLabel(1, label_b);
    HierarchicalHMM a(NONE, &ts);
    a.set_nbStates(3);
    a.train();
    a.set_covariance_mode(GaussianDistribution::DIAGONAL);
    a.performance_init();
    vector<double> log_likelihood(100, 0.0);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        a.performance_update(observation);
        log_likelihood[i] = a.results_log_likelihoods[0];
    }
    a.set_covariance_mode(GaussianDistribution::FULL);
    a.performance_init();
    vector<double> log_likelihood2(100, 0.0);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        a.performance_update(observation);
        log_likelihood2[i] = a.results_log_likelihoods[0];
    }
    CHECK_VECTOR_APPROX(log_likelihood, log_likelihood2);
    CHECK_NOTHROW(a.set_covariance_mode(GaussianDistribution::DIAGONAL));
    JSONNode a_json = a.to_json();
    HierarchicalHMM b;
    b.from_json(a_json);
}

//TEST_CASE( "Computation Time with Diagonal covariance", "[GaussianDistribution]" ) {
//    unsigned int dimension(300);
//    GaussianDistribution a(NONE, dimension);
//    for (unsigned int i=0; i<dimension; ++i) {
//        a.mean[i] = double(rand()) / RAND_MAX;
//        for (unsigned int j=0; j<dimension; ++j)
//            a.covariance[i*dimension+j] = double(rand()) / RAND_MAX;
//    }
//    a.addOffset();
//    a.updateInverseCovariance();
//    float* observation = new float[dimension];
//    for (unsigned int i=0; i<dimension; ++i) {
//        observation[i] = double(rand()) / RAND_MAX;
//    }
//    unsigned int num_iterations = 10000;
//    clock_t begin = clock();
//    for (unsigned int t=0; t<num_iterations; ++t) {
//        a.likelihood(observation);
//    }
//    clock_t end = clock();
//    double elapsed_ms = 1000. * double(end - begin) / CLOCKS_PER_SEC;
//    cout << "Evalution time (full covariance) = " << elapsed_ms << endl;
//    
//    a.set_covariance_mode(GaussianDistribution::DIAGONAL);
//    begin = clock();
//    for (unsigned int t=0; t<num_iterations; ++t) {
//        a.likelihood(observation);
//    }
//    end = clock();
//    elapsed_ms = 1000. * double(end - begin) / CLOCKS_PER_SEC;
//    cout << "Evalution time (Diagonal covariance) = " << elapsed_ms << endl;
//    
//    delete [] observation;
//    
//    dimension = 30;
//    TrainingSet ts(NONE, dimension);
//    vector<float> observation_(dimension);
//    for (unsigned int i=0; i<100; i++) {
//        for (unsigned int d=0; d<dimension; ++d) {
//            observation_[d] = float(rand()) / RAND_MAX;
//        }
//        ts.recordPhrase(0, observation_);
//    }
//    HMM hmm(NONE, &ts);
//    hmm.set_nbStates(30);
//    begin = clock();
//    hmm.train();
//    end = clock();
//    elapsed_ms = 1000. * double(end - begin) / CLOCKS_PER_SEC;
//    cout << "HMM training time (full covariance) = " << elapsed_ms << endl;
//    
//    hmm.set_covariance_mode(GaussianDistribution::DIAGONAL);
//    begin = clock();
//    hmm.train();
//    end = clock();
//    elapsed_ms = 1000. * double(end - begin) / CLOCKS_PER_SEC;
//    cout << "HMM training time (Diagonal covariance) = " << elapsed_ms << endl;
//}
