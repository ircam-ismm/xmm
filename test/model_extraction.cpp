/*
 * model_extraction.cpp
 *
 * Test suite for the extraction (submodel, reverse model)
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

TEST_CASE( "Extract Gaussian Distribution: arbitrary dimensions", "[GaussianDistribution]" ) {
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
    vector<unsigned int> columns(2);
    columns[0] = 2;
    columns[1] = 0;
    GaussianDistribution b = a.extract_submodel(columns);
    CHECK_FALSE(b.bimodal_);
    CHECK(b.dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.5;
    new_covariance[1] = 0.2;
    new_covariance[2] = 0.2;
    new_covariance[3] = 1.3;
    CHECK_VECTOR_APPROX(b.covariance, new_covariance);
    CHECK(b.mean[0] == Approx(a.mean[2]));
    CHECK(b.mean[1] == Approx(a.mean[0]));
}


TEST_CASE( "Extract Gaussian Distribution: input & output", "[GaussianDistribution]" ) {
    GaussianDistribution a(BIMODAL,
                           3,
                           2,
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
    GaussianDistribution b = a.extract_submodel_input();
    CHECK_FALSE(b.bimodal_);
    CHECK(b.dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.3;
    new_covariance[1] = 0.;
    new_covariance[2] = 0.;
    new_covariance[3] = 1.4;
    CHECK_VECTOR_APPROX(b.covariance, new_covariance);
    CHECK(b.mean[0] == Approx(a.mean[0]));
    CHECK(b.mean[1] == Approx(a.mean[1]));
    GaussianDistribution c = a.extract_submodel_output();
    CHECK_FALSE(c.bimodal_);
    CHECK(c.dimension_ == 1);
    new_covariance.resize(1);
    new_covariance[0] = 1.5;
    CHECK_VECTOR_APPROX(c.covariance, new_covariance);
    CHECK(c.mean[0] == Approx(a.mean[2]));
    CHECK(c.mean.size() == 1);
}

TEST_CASE( "Extract Gaussian Distribution: inverse model", "[GaussianDistribution]" ) {
    GaussianDistribution a(BIMODAL,
                           3,
                           2,
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
    GaussianDistribution b = a.extract_inverse_model();
    CHECK(b.bimodal_);
    CHECK(b.dimension_ == 3);
    vector<double> new_covariance(9);
    new_covariance[0] = 1.5;
    new_covariance[1] = 0.2;
    new_covariance[2] = 0.7;
    new_covariance[3] = 0.2;
    new_covariance[4] = 1.3;
    new_covariance[5] = 0.;
    new_covariance[6] = 0.7;
    new_covariance[7] = 0.;
    new_covariance[8] = 1.4;
    CHECK_VECTOR_APPROX(b.covariance, new_covariance);
    CHECK(b.mean[0] == Approx(a.mean[2]));
    CHECK(b.mean[1] == Approx(a.mean[0]));
    CHECK(b.mean[2] == Approx(a.mean[1]));
}


TEST_CASE( "Extract GMM: arbitrary dimensions", "[GMM]" ) {
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
    vector<float> mixtureCoeffs(3);
    mixtureCoeffs[0] = 3.819443583488e-01;
    mixtureCoeffs[1] = 3.171715140343e-01;
    mixtureCoeffs[2] = 3.008841276169e-01;
    CHECK_VECTOR_APPROX(a.mixtureCoeffs, mixtureCoeffs);
    vector<double> cov_c0(9);
    cov_c0[0] = 1.398498568044e-02;
    cov_c0[1] = 5.041130937517e-03;
    cov_c0[2] = 1.792829737244e-03;
    cov_c0[3] = 5.041130937517e-03;
    cov_c0[4] = 3.113048333532e-03;
    cov_c0[5] = 7.901977752380e-04;
    cov_c0[6] = 1.792829737244e-03;
    cov_c0[7] = 7.901977752380e-04;
    cov_c0[8] = 1.306463534393e-03;
    vector<unsigned int> columns(2);
    columns[0] = 2;
    columns[1] = 0;
    GMM b = a.extract_submodel(columns);
    CHECK_FALSE(b.bimodal_);
    CHECK(b.dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.306463534393e-03;
    new_covariance[1] = 1.792829737244e-03;
    new_covariance[2] = 1.792829737244e-03;
    new_covariance[3] = 1.398498568044e-02;
    CHECK_VECTOR_APPROX(b.components[0].covariance, new_covariance);
    CHECK_VECTOR_APPROX(b.mixtureCoeffs, a.mixtureCoeffs);
    CHECK(b.components[0].mean[0] == Approx(a.components[0].mean[2]));
    CHECK(b.components[0].mean[1] == Approx(a.components[0].mean[0]));
}


TEST_CASE( "Extract GMM: input & output", "[GMM]" ) {
    TrainingSet ts(BIMODAL, 3, 2);
    vector<float> observation(3);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.recordPhrase(0, observation);
    }
    GMM a(BIMODAL, &ts);
    a.set_nbMixtureComponents(3);
    a.train();
    vector<float> mixtureCoeffs(3);
    mixtureCoeffs[0] = 3.819443583488e-01;
    mixtureCoeffs[1] = 3.171715140343e-01;
    mixtureCoeffs[2] = 3.008841276169e-01;
    CHECK_VECTOR_APPROX(a.mixtureCoeffs, mixtureCoeffs);
    vector<double> cov_c0(9);
    cov_c0[0] = 1.398498568044e-02;
    cov_c0[1] = 5.041130937517e-03;
    cov_c0[2] = 1.792829737244e-03;
    cov_c0[3] = 5.041130937517e-03;
    cov_c0[4] = 3.113048333532e-03;
    cov_c0[5] = 7.901977752380e-04;
    cov_c0[6] = 1.792829737244e-03;
    cov_c0[7] = 7.901977752380e-04;
    cov_c0[8] = 1.306463534393e-03;
    GMM b = a.extract_submodel_input();
    CHECK_FALSE(b.bimodal_);
    CHECK(b.dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.398498568044e-02;
    new_covariance[1] = 5.041130937517e-03;
    new_covariance[2] = 5.041130937517e-03;
    new_covariance[3] = 3.113048333532e-03;
    CHECK_VECTOR_APPROX(b.components[0].covariance, new_covariance);
    CHECK(b.components[0].mean[0] == Approx(a.components[0].mean[0]));
    CHECK(b.components[0].mean[1] == Approx(a.components[0].mean[1]));
    GMM c = a.extract_submodel_output();
    CHECK_FALSE(c.bimodal_);
    CHECK(c.dimension_ == 1);
    new_covariance.resize(1);
    new_covariance[0] = 1.306463534393e-03;
    CHECK_VECTOR_APPROX(c.components[0].covariance, new_covariance);
    CHECK(c.components[0].mean[0] == Approx(a.components[0].mean[2]));
    CHECK(c.components[0].mean.size() == 1);
}

TEST_CASE( "Extract GMM: inverse model", "[GMM]" ) {
    TrainingSet ts(BIMODAL, 3, 2);
    vector<float> observation(3);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.recordPhrase(0, observation);
    }
    GMM a(BIMODAL, &ts);
    a.set_nbMixtureComponents(3);
    a.train();
    vector<float> mixtureCoeffs(3);
    mixtureCoeffs[0] = 3.819443583488e-01;
    mixtureCoeffs[1] = 3.171715140343e-01;
    mixtureCoeffs[2] = 3.008841276169e-01;
    CHECK_VECTOR_APPROX(a.mixtureCoeffs, mixtureCoeffs);
    vector<double> cov_c0(9);
    cov_c0[0] = 1.398498568044e-02;
    cov_c0[1] = 5.041130937517e-03;
    cov_c0[2] = 1.792829737244e-03;
    cov_c0[3] = 5.041130937517e-03;
    cov_c0[4] = 3.113048333532e-03;
    cov_c0[5] = 7.901977752380e-04;
    cov_c0[6] = 1.792829737244e-03;
    cov_c0[7] = 7.901977752380e-04;
    cov_c0[8] = 1.306463534393e-03;
    GMM b = a.extract_inverse_model();
    CHECK(b.bimodal_);
    CHECK(b.dimension_ == 3);
    vector<double> new_covariance(9);
    new_covariance[0] = 1.306463534393e-03;
    new_covariance[1] = 1.792829737244e-03;
    new_covariance[2] = 7.901977752380e-04;
    new_covariance[3] = 1.792829737244e-03;
    new_covariance[4] = 1.398498568044e-02;
    new_covariance[5] = 5.041130937517e-03;
    new_covariance[6] = 7.901977752380e-04;
    new_covariance[7] = 5.041130937517e-03;
    new_covariance[8] = 3.113048333532e-03;
    CHECK_VECTOR_APPROX(b.components[0].covariance, new_covariance);
    CHECK(b.components[0].mean[0] == Approx(a.components[0].mean[2]));
    CHECK(b.components[0].mean[1] == Approx(a.components[0].mean[0]));
    CHECK(b.components[0].mean[2] == Approx(a.components[0].mean[1]));
}

TEST_CASE( "Extract GMMGroup: arbitrary dimensions", "[GMMGroup]" ) {
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
    vector<float> mixtureCoeffs(3);
    mixtureCoeffs[0] = 3.819443583488e-01;
    mixtureCoeffs[1] = 3.171715140343e-01;
    mixtureCoeffs[2] = 3.008841276169e-01;
    CHECK_VECTOR_APPROX(a.models[label_a].mixtureCoeffs, mixtureCoeffs);
    vector<double> cov_c0(9);
    cov_c0[0] = 1.398498568044e-02;
    cov_c0[1] = 5.041130937517e-03;
    cov_c0[2] = 1.792829737244e-03;
    cov_c0[3] = 5.041130937517e-03;
    cov_c0[4] = 3.113048333532e-03;
    cov_c0[5] = 7.901977752380e-04;
    cov_c0[6] = 1.792829737244e-03;
    cov_c0[7] = 7.901977752380e-04;
    cov_c0[8] = 1.306463534393e-03;
    vector<unsigned int> columns(2);
    columns[0] = 2;
    columns[1] = 0;
    GMMGroup b = a.extract_submodel(columns);
    CHECK_FALSE(b.bimodal_);
    CHECK(b.models[label_a].dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.306463534393e-03;
    new_covariance[1] = 1.792829737244e-03;
    new_covariance[2] = 1.792829737244e-03;
    new_covariance[3] = 1.398498568044e-02;
    CHECK_VECTOR_APPROX(b.models[label_a].components[0].covariance, new_covariance);
    CHECK_VECTOR_APPROX(b.models[label_a].mixtureCoeffs, a.models[label_a].mixtureCoeffs);
    CHECK(b.models[label_a].components[0].mean[0] == Approx(a.models[label_a].components[0].mean[2]));
    CHECK(b.models[label_a].components[0].mean[1] == Approx(a.models[label_a].components[0].mean[0]));
    CHECK_VECTOR_APPROX(b.models[label_b].components[0].covariance, new_covariance);
    CHECK_VECTOR_APPROX(b.models[label_b].mixtureCoeffs, a.models[label_b].mixtureCoeffs);
    CHECK(b.models[label_b].components[0].mean[0] == Approx(a.models[label_b].components[0].mean[2]));
    CHECK(b.models[label_b].components[0].mean[1] == Approx(a.models[label_b].components[0].mean[0]));
}


TEST_CASE( "Extract GMMGroup: input & output", "[GMMGroup]" ) {
    TrainingSet ts(BIMODAL, 3, 2);
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
    GMMGroup a(BIMODAL, &ts);
    a.set_nbMixtureComponents(3);
    a.train();
    vector<float> mixtureCoeffs(3);
    mixtureCoeffs[0] = 3.819443583488e-01;
    mixtureCoeffs[1] = 3.171715140343e-01;
    mixtureCoeffs[2] = 3.008841276169e-01;
    CHECK_VECTOR_APPROX(a.models[label_a].mixtureCoeffs, mixtureCoeffs);
    vector<double> cov_c0(9);
    cov_c0[0] = 1.398498568044e-02;
    cov_c0[1] = 5.041130937517e-03;
    cov_c0[2] = 1.792829737244e-03;
    cov_c0[3] = 5.041130937517e-03;
    cov_c0[4] = 3.113048333532e-03;
    cov_c0[5] = 7.901977752380e-04;
    cov_c0[6] = 1.792829737244e-03;
    cov_c0[7] = 7.901977752380e-04;
    cov_c0[8] = 1.306463534393e-03;
    GMMGroup b = a.extract_submodel_input();
    CHECK_FALSE(b.bimodal_);
    CHECK(b.models[label_a].dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.398498568044e-02;
    new_covariance[1] = 5.041130937517e-03;
    new_covariance[2] = 5.041130937517e-03;
    new_covariance[3] = 3.113048333532e-03;
    CHECK_VECTOR_APPROX(b.models[label_a].components[0].covariance, new_covariance);
    CHECK_VECTOR_APPROX(b.models[label_b].components[0].covariance, new_covariance);
    CHECK(b.models[label_a].components[0].mean[0] == Approx(a.models[label_a].components[0].mean[0]));
    CHECK(b.models[label_a].components[0].mean[1] == Approx(a.models[label_a].components[0].mean[1]));
    GMMGroup c = a.extract_submodel_output();
    CHECK_FALSE(c.bimodal_);
    CHECK(c.models[label_a].dimension_ == 1);
    new_covariance.resize(1);
    new_covariance[0] = 1.306463534393e-03;
    CHECK_VECTOR_APPROX(c.models[label_a].components[0].covariance, new_covariance);
    CHECK(c.models[label_a].components[0].mean[0] == Approx(a.models[label_a].components[0].mean[2]));
    CHECK(c.models[label_a].components[0].mean.size() == 1);

}

TEST_CASE( "Extract GMMGroup: inverse model", "[GMMGroup]" ) {
    TrainingSet ts(BIMODAL, 3, 2);
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
    GMMGroup a(BIMODAL, &ts);
    a.set_nbMixtureComponents(3);
    a.train();
    vector<float> mixtureCoeffs(3);
    mixtureCoeffs[0] = 3.819443583488e-01;
    mixtureCoeffs[1] = 3.171715140343e-01;
    mixtureCoeffs[2] = 3.008841276169e-01;
    CHECK_VECTOR_APPROX(a.models[label_a].mixtureCoeffs, mixtureCoeffs);
    vector<double> cov_c0(9);
    cov_c0[0] = 1.398498568044e-02;
    cov_c0[1] = 5.041130937517e-03;
    cov_c0[2] = 1.792829737244e-03;
    cov_c0[3] = 5.041130937517e-03;
    cov_c0[4] = 3.113048333532e-03;
    cov_c0[5] = 7.901977752380e-04;
    cov_c0[6] = 1.792829737244e-03;
    cov_c0[7] = 7.901977752380e-04;
    cov_c0[8] = 1.306463534393e-03;
    GMMGroup b = a.extract_inverse_model();
    CHECK(b.bimodal_);
    CHECK(b.models[label_a].dimension_ == 3);
    vector<double> new_covariance(9);
    new_covariance[0] = 1.306463534393e-03;
    new_covariance[1] = 1.792829737244e-03;
    new_covariance[2] = 7.901977752380e-04;
    new_covariance[3] = 1.792829737244e-03;
    new_covariance[4] = 1.398498568044e-02;
    new_covariance[5] = 5.041130937517e-03;
    new_covariance[6] = 7.901977752380e-04;
    new_covariance[7] = 5.041130937517e-03;
    new_covariance[8] = 3.113048333532e-03;
    CHECK_VECTOR_APPROX(b.models[label_a].components[0].covariance, new_covariance);
    CHECK(b.models[label_a].components[0].mean[0] == Approx(a.models[label_a].components[0].mean[2]));
    CHECK(b.models[label_a].components[0].mean[1] == Approx(a.models[label_a].components[0].mean[0]));
    CHECK(b.models[label_a].components[0].mean[2] == Approx(a.models[label_a].components[0].mean[1]));
    CHECK_VECTOR_APPROX(b.models[label_b].components[0].covariance, new_covariance);
    CHECK(b.models[label_b].components[0].mean[0] == Approx(a.models[label_b].components[0].mean[2]));
    CHECK(b.models[label_b].components[0].mean[1] == Approx(a.models[label_b].components[0].mean[0]));
    CHECK(b.models[label_b].components[0].mean[2] == Approx(a.models[label_b].components[0].mean[1]));
}

TEST_CASE( "Extract HMM: arbitrary dimensions", "[HMM]" ) {
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
    vector<float> transition(6);
    transition[0] = 9.710567593575e-01;
    transition[1] = 2.894317358732e-02;
    transition[2] = 9.693398475647e-01;
    transition[3] = 3.066014684737e-02;
    transition[4] = 1;
    transition[5] = 0;
    CHECK_VECTOR_APPROX(a.transition, transition);
    vector<double> covariance_state0(9);
    covariance_state0[0] = 1.105028519640e-02,
    covariance_state0[1] = 3.397482636225e-03;
    covariance_state0[2] = 1.045622665096e-03;
    covariance_state0[3] = 3.397482636225e-03;
    covariance_state0[4] = 2.231733755155e-03;
    covariance_state0[5] = 3.960464278701e-04;
    covariance_state0[6] = 1.045622665096e-03;
    covariance_state0[7] = 3.960464278701e-04;
    covariance_state0[8] = 1.131260901171e-03;
    CHECK_VECTOR_APPROX(a.states[0].components[0].covariance, covariance_state0);
    CHECK_FALSE(a.bimodal_);
    CHECK(a.states[0].components[0].inverseCovariance_input_.size() == 0);
    vector<unsigned int> columns(2);
    columns[0] = 2;
    columns[1] = 0;
    HMM b = a.extract_submodel(columns);
    CHECK_FALSE(b.bimodal_);
    CHECK(b.states[0].dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.131260901171e-03;
    new_covariance[1] = 1.045622665096e-03;
    new_covariance[2] = 1.045622665096e-03;
    new_covariance[3] = 1.105028519640e-02;
    CHECK_VECTOR_APPROX(b.states[0].components[0].covariance, new_covariance);
    CHECK_VECTOR_APPROX(b.states[0].mixtureCoeffs, a.states[0].mixtureCoeffs);
    CHECK(b.states[0].components[0].mean[0] == Approx(a.states[0].components[0].mean[2]));
    CHECK(b.states[0].components[0].mean[1] == Approx(a.states[0].components[0].mean[0]));
}


TEST_CASE( "Extract HMM: input & output", "[HMM]" ) {
    TrainingSet ts(BIMODAL, 3, 2);
    vector<float> observation(3);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.recordPhrase(0, observation);
    }
    HMM a(BIMODAL, &ts);
    a.set_nbStates(3);
    a.train();
    vector<float> transition(6);
    transition[0] = 9.710567593575e-01;
    transition[1] = 2.894317358732e-02;
    transition[2] = 9.693398475647e-01;
    transition[3] = 3.066014684737e-02;
    transition[4] = 1;
    transition[5] = 0;
    CHECK_VECTOR_APPROX(a.transition, transition);
    vector<double> covariance_state0(9);
    covariance_state0[0] = 1.105028519640e-02,
    covariance_state0[1] = 3.397482636225e-03;
    covariance_state0[2] = 1.045622665096e-03;
    covariance_state0[3] = 3.397482636225e-03;
    covariance_state0[4] = 2.231733755155e-03;
    covariance_state0[5] = 3.960464278701e-04;
    covariance_state0[6] = 1.045622665096e-03;
    covariance_state0[7] = 3.960464278701e-04;
    covariance_state0[8] = 1.131260901171e-03;
    CHECK_VECTOR_APPROX(a.states[0].components[0].covariance, covariance_state0);
    CHECK(a.bimodal_);
    HMM b = a.extract_submodel_input();
    CHECK_FALSE(b.bimodal_);
    CHECK(b.states[0].dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.105028519640e-02;
    new_covariance[1] = 3.397482636225e-03;
    new_covariance[2] = 3.397482636225e-03;
    new_covariance[3] = 2.231733755155e-03;
    CHECK_VECTOR_APPROX(b.states[0].components[0].covariance, new_covariance);
    CHECK_VECTOR_APPROX(b.states[0].mixtureCoeffs, a.states[0].mixtureCoeffs);
    CHECK(b.states[0].components[0].mean[0] == Approx(a.states[0].components[0].mean[0]));
    CHECK(b.states[0].components[0].mean[1] == Approx(a.states[0].components[0].mean[1]));
    HMM c = a.extract_submodel_output();
    CHECK_FALSE(c.bimodal_);
    CHECK(c.states[0].dimension_ == 1);
    vector<double> new_covariance_output(1);
    new_covariance_output[0] = 1.131260901171e-03;
    CHECK_VECTOR_APPROX(c.states[0].components[0].covariance, new_covariance_output);
    CHECK_VECTOR_APPROX(c.states[0].mixtureCoeffs, a.states[0].mixtureCoeffs);
    CHECK(c.states[0].components[0].mean[0] == Approx(a.states[0].components[0].mean[2]));
}

TEST_CASE( "Extract HMM: inverse model", "[HMM]" ) {
    TrainingSet ts(BIMODAL, 3, 2);
    vector<float> observation(3);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.recordPhrase(0, observation);
    }
    HMM a(BIMODAL, &ts);
    a.set_nbStates(3);
    a.train();
    vector<float> transition(6);
    transition[0] = 9.710567593575e-01;
    transition[1] = 2.894317358732e-02;
    transition[2] = 9.693398475647e-01;
    transition[3] = 3.066014684737e-02;
    transition[4] = 1;
    transition[5] = 0;
    CHECK_VECTOR_APPROX(a.transition, transition);
    vector<double> covariance_state0(9);
    covariance_state0[0] = 1.105028519640e-02,
    covariance_state0[1] = 3.397482636225e-03;
    covariance_state0[2] = 1.045622665096e-03;
    covariance_state0[3] = 3.397482636225e-03;
    covariance_state0[4] = 2.231733755155e-03;
    covariance_state0[5] = 3.960464278701e-04;
    covariance_state0[6] = 1.045622665096e-03;
    covariance_state0[7] = 3.960464278701e-04;
    covariance_state0[8] = 1.131260901171e-03;
    CHECK_VECTOR_APPROX(a.states[0].components[0].covariance, covariance_state0);
    CHECK(a.bimodal_);
    CHECK(a.dimension_input_ == 2);
    HMM b = a.extract_inverse_model();
    CHECK(b.bimodal_);
    CHECK(b.dimension_ == 3);
    CHECK(b.dimension_input_ == 1);
    vector<double> new_covariance(9);
    new_covariance[0] = 1.131260901171e-03;
    new_covariance[1] = 1.045622665096e-03;
    new_covariance[2] = 3.960464278701e-04;
    new_covariance[3] = 1.045622665096e-03;
    new_covariance[4] = 1.105028519640e-02;
    new_covariance[5] = 3.397482636225e-03;
    new_covariance[6] = 3.960464278701e-04;
    new_covariance[7] = 3.397482636225e-03;
    new_covariance[8] = 2.231733755155e-03;
    CHECK_VECTOR_APPROX(b.states[0].components[0].covariance, new_covariance);
    CHECK(b.states[0].components[0].mean[0] == Approx(a.states[0].components[0].mean[2]));
    CHECK(b.states[0].components[0].mean[1] == Approx(a.states[0].components[0].mean[0]));
    CHECK(b.states[0].components[0].mean[2] == Approx(a.states[0].components[0].mean[1]));
    b.performance_init();
    observation.resize(1);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = pow(float(i)/100., 3.);
        b.performance_update(observation);
    }
}

TEST_CASE( "Extract HierarchicalHMM: arbitrary dimensions", "[HierarchicalHMM]" ) {
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
    vector<float> transition(6);
    transition[0] = 9.710567593575e-01;
    transition[1] = 2.894317358732e-02;
    transition[2] = 9.693398475647e-01;
    transition[3] = 3.066014684737e-02;
    transition[4] = 1;
    transition[5] = 0;
    CHECK_VECTOR_APPROX(a.models[label_a].transition, transition);
    vector<double> covariance_state0(9);
    covariance_state0[0] = 1.105028519640e-02,
    covariance_state0[1] = 3.397482636225e-03;
    covariance_state0[2] = 1.045622665096e-03;
    covariance_state0[3] = 3.397482636225e-03;
    covariance_state0[4] = 2.231733755155e-03;
    covariance_state0[5] = 3.960464278701e-04;
    covariance_state0[6] = 1.045622665096e-03;
    covariance_state0[7] = 3.960464278701e-04;
    covariance_state0[8] = 1.131260901171e-03;
    CHECK_VECTOR_APPROX(a.models[label_a].states[0].components[0].covariance, covariance_state0);
    CHECK_FALSE(a.bimodal_);
    CHECK(a.models[label_a].states[0].components[0].inverseCovariance_input_.size() == 0);
    vector<unsigned int> columns(2);
    columns[0] = 2;
    columns[1] = 0;
    HierarchicalHMM b = a.extract_submodel(columns);
    CHECK_FALSE(b.bimodal_);
    CHECK(b.models[label_a].states[0].dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.131260901171e-03;
    new_covariance[1] = 1.045622665096e-03;
    new_covariance[2] = 1.045622665096e-03;
    new_covariance[3] = 1.105028519640e-02;
    CHECK_VECTOR_APPROX(b.models[label_a].states[0].components[0].covariance, new_covariance);
    CHECK_VECTOR_APPROX(b.models[label_a].states[0].mixtureCoeffs, a.models[label_a].states[0].mixtureCoeffs);
    CHECK(b.models[label_a].states[0].components[0].mean[0] == Approx(a.models[label_a].states[0].components[0].mean[2]));
    CHECK(b.models[label_a].states[0].components[0].mean[1] == Approx(a.models[label_a].states[0].components[0].mean[0]));
}


TEST_CASE( "Extract HierarchicalHMM: input & output", "[HierarchicalHMM]" ) {
    TrainingSet ts(BIMODAL, 3, 2);
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
    HierarchicalHMM a(BIMODAL, &ts);
    a.set_nbStates(3);
    a.train();
    vector<float> transition(6);
    transition[0] = 9.710567593575e-01;
    transition[1] = 2.894317358732e-02;
    transition[2] = 9.693398475647e-01;
    transition[3] = 3.066014684737e-02;
    transition[4] = 1;
    transition[5] = 0;
    CHECK_VECTOR_APPROX(a.models[label_a].transition, transition);
    vector<double> covariance_state0(9);
    covariance_state0[0] = 1.105028519640e-02,
    covariance_state0[1] = 3.397482636225e-03;
    covariance_state0[2] = 1.045622665096e-03;
    covariance_state0[3] = 3.397482636225e-03;
    covariance_state0[4] = 2.231733755155e-03;
    covariance_state0[5] = 3.960464278701e-04;
    covariance_state0[6] = 1.045622665096e-03;
    covariance_state0[7] = 3.960464278701e-04;
    covariance_state0[8] = 1.131260901171e-03;
    CHECK_VECTOR_APPROX(a.models[label_a].states[0].components[0].covariance, covariance_state0);
    CHECK(a.bimodal_);
    HierarchicalHMM b = a.extract_submodel_input();
    CHECK_FALSE(b.bimodal_);
    CHECK(b.models[label_a].states[0].dimension_ == 2);
    vector<double> new_covariance(4);
    new_covariance[0] = 1.105028519640e-02;
    new_covariance[1] = 3.397482636225e-03;
    new_covariance[2] = 3.397482636225e-03;
    new_covariance[3] = 2.231733755155e-03;
    CHECK_VECTOR_APPROX(b.models[label_a].states[0].components[0].covariance, new_covariance);
    CHECK_VECTOR_APPROX(b.models[label_a].states[0].mixtureCoeffs, a.models[label_a].states[0].mixtureCoeffs);
    CHECK(b.models[label_a].states[0].components[0].mean[0] == Approx(a.models[label_a].states[0].components[0].mean[0]));
    CHECK(b.models[label_a].states[0].components[0].mean[1] == Approx(a.models[label_a].states[0].components[0].mean[1]));
    HierarchicalHMM c = a.extract_submodel_output();
    CHECK_FALSE(c.bimodal_);
    CHECK(c.models[label_a].states[0].dimension_ == 1);
    vector<double> new_covariance_output(1);
    new_covariance_output[0] = 1.131260901171e-03;
    CHECK_VECTOR_APPROX(c.models[label_a].states[0].components[0].covariance, new_covariance_output);
    CHECK_VECTOR_APPROX(c.models[label_a].states[0].mixtureCoeffs, a.models[label_a].states[0].mixtureCoeffs);
    CHECK(c.models[label_a].states[0].components[0].mean[0] == Approx(a.models[label_a].states[0].components[0].mean[2]));
}

TEST_CASE( "Extract HierarchicalHMM: inverse model", "[HierarchicalHMM]" ) {
    TrainingSet ts(BIMODAL, 3, 2);
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
    HierarchicalHMM a(BIMODAL, &ts);
    a.set_nbStates(3);
    a.train();
    vector<float> transition(6);
    transition[0] = 9.710567593575e-01;
    transition[1] = 2.894317358732e-02;
    transition[2] = 9.693398475647e-01;
    transition[3] = 3.066014684737e-02;
    transition[4] = 1;
    transition[5] = 0;
    CHECK_VECTOR_APPROX(a.models[label_a].transition, transition);
    vector<double> covariance_state0(9);
    covariance_state0[0] = 1.105028519640e-02,
    covariance_state0[1] = 3.397482636225e-03;
    covariance_state0[2] = 1.045622665096e-03;
    covariance_state0[3] = 3.397482636225e-03;
    covariance_state0[4] = 2.231733755155e-03;
    covariance_state0[5] = 3.960464278701e-04;
    covariance_state0[6] = 1.045622665096e-03;
    covariance_state0[7] = 3.960464278701e-04;
    covariance_state0[8] = 1.131260901171e-03;
    CHECK_VECTOR_APPROX(a.models[label_a].states[0].components[0].covariance, covariance_state0);
    CHECK(a.bimodal_);
    CHECK(a.models[label_a].dimension_input_ == 2);
    HierarchicalHMM b = a.extract_inverse_model();
    CHECK(b.bimodal_);
    CHECK(b.models[label_a].dimension_ == 3);
    CHECK(b.models[label_a].dimension_input_ == 1);
    vector<double> new_covariance(9);
    new_covariance[0] = 1.131260901171e-03;
    new_covariance[1] = 1.045622665096e-03;
    new_covariance[2] = 3.960464278701e-04;
    new_covariance[3] = 1.045622665096e-03;
    new_covariance[4] = 1.105028519640e-02;
    new_covariance[5] = 3.397482636225e-03;
    new_covariance[6] = 3.960464278701e-04;
    new_covariance[7] = 3.397482636225e-03;
    new_covariance[8] = 2.231733755155e-03;
    CHECK_VECTOR_APPROX(b.models[label_a].states[0].components[0].covariance, new_covariance);
    CHECK(b.models[label_a].states[0].components[0].mean[0] == Approx(a.models[label_a].states[0].components[0].mean[2]));
    CHECK(b.models[label_a].states[0].components[0].mean[1] == Approx(a.models[label_a].states[0].components[0].mean[0]));
    CHECK(b.models[label_a].states[0].components[0].mean[2] == Approx(a.models[label_a].states[0].components[0].mean[1]));
    b.performance_init();
    observation.resize(1);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = pow(float(i)/100., 3.);
        b.performance_update(observation);
    }
}

