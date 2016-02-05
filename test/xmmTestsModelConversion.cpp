/*
 * xmmTestsModelConversion.cpp
 *
 * Test suite for the conversion (unimodal <-> bimodal)
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
#include "xmmTestsUtilities.hpp"
#define XMM_TESTING
#include "xmm.h"

//TEST_CASE( "Convert Gaussian Distribution: unimodal->bimodal", "[GaussianDistribution]" ) {
//    xmm::GaussianDistribution a(false,
//                           3,
//                           0,
//                           0.0034,
//                           0.0123);
//    a.mean[0] = 0.2;
//    a.mean[1] = 0.3;
//    a.mean[2] = 0.1;
//    a.covariance[0] = 1.3;
//    a.covariance[1] = 0.0;
//    a.covariance[2] = 0.2;
//    a.covariance[3] = 0.0;
//    a.covariance[4] = 1.4;
//    a.covariance[5] = 0.7;
//    a.covariance[6] = 0.2;
//    a.covariance[7] = 0.7;
//    a.covariance[8] = 1.5;
//    a.updateInverseCovariance();
//    xmm::GaussianDistribution b(a);
//    CHECK_THROWS(b.makeUnimodal());
//    CHECK_NOTHROW(b.makeBimodal(2));
//    CHECK(b.bimodal_);
//    CHECK_FALSE(a.bimodal_);
//    CHECK(a.dimension.get() == 3);
//    CHECK(a.dimension_input.get() == 0);
//    CHECK(b.dimension.get() == 3);
//    CHECK(b.dimension_input.get() == 2);
//    CHECK(a.mean == b.mean);
//    CHECK(a.covariance == b.covariance);
//    CHECK(a.inverseCovariance_ == b.inverseCovariance_);
//    std::vector<double> invcov(9);
//    invcov[0] = 0.79037801;
//    invcov[1] = 0.06872852;
//    invcov[2] = -0.13745704;
//    invcov[3] = 0.06872852;
//    invcov[4] = 0.93765341;
//    invcov[5] = -0.4467354;
//    invcov[6] = -0.13745704;
//    invcov[7] = -0.4467354;
//    invcov[8] = 0.89347079;
//    CHECK_VECTOR_APPROX(b.inverseCovariance_, invcov);
//    std::vector<double> invcov_input(4);
//    invcov_input[0] = 0.76923077;
//    invcov_input[1] = 0.;
//    invcov_input[2] = 0.;
//    invcov_input[3] = 0.71428571;
//    CHECK(a.inverseCovariance_input_.empty());
//    CHECK_VECTOR_APPROX(b.inverseCovariance_input_, invcov_input);
//}
//
//TEST_CASE( "Convert Gaussian Distribution: bimodal->unimodal", "[GaussianDistribution]" ) {
//    xmm::GaussianDistribution a(true,
//                           3,
//                           2,
//                           0.0034,
//                           0.0123);
//    a.mean[0] = 0.2;
//    a.mean[1] = 0.3;
//    a.mean[2] = 0.1;
//    a.covariance[0] = 1.3;
//    a.covariance[1] = 0.0;
//    a.covariance[2] = 0.2;
//    a.covariance[3] = 0.0;
//    a.covariance[4] = 1.4;
//    a.covariance[5] = 0.7;
//    a.covariance[6] = 0.2;
//    a.covariance[7] = 0.7;
//    a.covariance[8] = 1.5;
//    a.updateInverseCovariance();
//    xmm::GaussianDistribution b(a);
//    CHECK_THROWS(b.makeBimodal(2));
//    CHECK_NOTHROW(b.makeUnimodal());
//    CHECK(a.bimodal_);
//    CHECK_FALSE(b.bimodal_);
//    CHECK(a.dimension.get() == 3);
//    CHECK(a.dimension_input.get() == 2);
//    CHECK(b.dimension.get() == 3);
//    CHECK(b.dimension_input.get() == 0);
//    CHECK(a.mean == b.mean);
//    CHECK(a.covariance == b.covariance);
//    CHECK(a.inverseCovariance_ == b.inverseCovariance_);
//    std::vector<double> invcov(9);
//    invcov[0] = 0.79037801;
//    invcov[1] = 0.06872852;
//    invcov[2] = -0.13745704;
//    invcov[3] = 0.06872852;
//    invcov[4] = 0.93765341;
//    invcov[5] = -0.4467354;
//    invcov[6] = -0.13745704;
//    invcov[7] = -0.4467354;
//    invcov[8] = 0.89347079;
//    CHECK_VECTOR_APPROX(b.inverseCovariance_, invcov);
//    std::vector<double> invcov_input(4);
//    invcov_input[0] = 0.76923077;
//    invcov_input[1] = 0.;
//    invcov_input[2] = 0.;
//    invcov_input[3] = 0.71428571;
//    CHECK(b.inverseCovariance_input_.empty());
//    CHECK_VECTOR_APPROX(a.inverseCovariance_input_, invcov_input);
//    
//}
//
//TEST_CASE( "Convert GMM: unimodal->bimodal", "[GMM]" ) {
//    xmm::TrainingSet ts(true, false, 3);
//    std::vector<float> observation(3);
//    ts.addPhrase(0);
//    for (std::size_t i=0; i<100; i++) {
//        observation[0] = float(i)/100.;
//        observation[1] = pow(float(i)/100., 2.);
//        observation[2] = pow(float(i)/100., 3.);
//        ts.getPhrase(0)->record(observation);
//    }
//    xmm::GMM a(xmm::NONE);
//    a.set_nbMixtureComponents(3);
//    a.train(&ts);
//    std::vector<float> mixtureCoeffs(3);
//    mixtureCoeffs[0] = 3.819443583488e-01;
//    mixtureCoeffs[1] = 3.171715140343e-01;
//    mixtureCoeffs[2] = 3.008841276169e-01;
//    CHECK_VECTOR_APPROX(a.mixtureCoeffs, mixtureCoeffs);
//    std::vector<double> cov_c0(9);
//    cov_c0[0] = 1.398498568044e-02;
//    cov_c0[1] = 5.041130937517e-03;
//    cov_c0[2] = 1.792829737244e-03;
//    cov_c0[3] = 5.041130937517e-03;
//    cov_c0[4] = 3.113048333532e-03;
//    cov_c0[5] = 7.901977752380e-04;
//    cov_c0[6] = 1.792829737244e-03;
//    cov_c0[7] = 7.901977752380e-04;
//    cov_c0[8] = 1.306463534393e-03;
//    CHECK_VECTOR_APPROX(a.components[0].covariance, cov_c0);
//    CHECK_FALSE(a.bimodal_);
//    CHECK(a.components[0].inverseCovariance_input_.size() == 0);
//    xmm::GMM b(a);
//    b.makeBimodal(2);
//    CHECK(b.bimodal_);
//    CHECK(a.mixtureCoeffs == b.mixtureCoeffs);
//    for (std::size_t i=0; i<3; i++) {
//        CHECK(a.components[i].mean == b.components[i].mean);
//        CHECK(a.components[i].covariance == b.components[i].covariance);
//        CHECK(a.components[i].inverseCovariance_ == b.components[i].inverseCovariance_);
//        CHECK(a.components[i].inverseCovariance_input_.size() == 0);
//        CHECK(b.components[i].inverseCovariance_input_.size() == 4);
//    }
//    std::vector<double> invcov_c0_input(4);
//    invcov_c0_input[0] = 171.77395895;
//    invcov_c0_input[1] = -278.16304982;
//    invcov_c0_input[2] = -278.16304982;
//    invcov_c0_input[3] = 771.67332427;
//    CHECK_VECTOR_APPROX(b.components[0].inverseCovariance_input_, invcov_c0_input);
//}
//
//TEST_CASE( "Convert GMM: bimodal->unimodal", "[GMM]" ) {
//    xmm::TrainingSet ts(xmm::BIMODAL, 3, 2);
//    std::vector<float> observation(3);
//    ts.addPhrase(0);
//    for (std::size_t i=0; i<100; i++) {
//        observation[0] = float(i)/100.;
//        observation[1] = pow(float(i)/100., 2.);
//        observation[2] = pow(float(i)/100., 3.);
//        ts.getPhrase(0)->record(observation);
//    }
//    xmm::GMM a(xmm::BIMODAL);
//    a.set_nbMixtureComponents(3);
//    a.train(&ts);
//    std::vector<float> mixtureCoeffs(3);
//    mixtureCoeffs[0] = 3.819443583488e-01;
//    mixtureCoeffs[1] = 3.171715140343e-01;
//    mixtureCoeffs[2] = 3.008841276169e-01;
//    CHECK_VECTOR_APPROX(a.mixtureCoeffs, mixtureCoeffs);
//    std::vector<double> cov_c0(9);
//    cov_c0[0] = 1.398498568044e-02;
//    cov_c0[1] = 5.041130937517e-03;
//    cov_c0[2] = 1.792829737244e-03;
//    cov_c0[3] = 5.041130937517e-03;
//    cov_c0[4] = 3.113048333532e-03;
//    cov_c0[5] = 7.901977752380e-04;
//    cov_c0[6] = 1.792829737244e-03;
//    cov_c0[7] = 7.901977752380e-04;
//    cov_c0[8] = 1.306463534393e-03;
//    CHECK_VECTOR_APPROX(a.components[0].covariance, cov_c0);
//    CHECK(a.bimodal_);
//    xmm::GMM b(a);
//    CHECK_THROWS(b.makeBimodal(2));
//    CHECK_NOTHROW(b.makeUnimodal());
//    CHECK_FALSE(b.bimodal_);
//    CHECK(a.mixtureCoeffs == b.mixtureCoeffs);
//    for (std::size_t i=0; i<3; i++) {
//        CHECK(a.components[i].mean == b.components[i].mean);
//        CHECK(a.components[i].covariance == b.components[i].covariance);
//        CHECK(a.components[i].inverseCovariance_ == b.components[i].inverseCovariance_);
//        CHECK(b.components[i].inverseCovariance_input_.size() == 0);
//        CHECK(a.components[i].inverseCovariance_input_.size() == 4);
//    }
//    std::vector<double> invcov_c0_input(4);
//    invcov_c0_input[0] = 171.77395895;
//    invcov_c0_input[1] = -278.16304982;
//    invcov_c0_input[2] = -278.16304982;
//    invcov_c0_input[3] = 771.67332427;
//    CHECK_VECTOR_APPROX(a.components[0].inverseCovariance_input_, invcov_c0_input);
//}
//
//TEST_CASE( "Convert GMM: unimodal->bimodal", "[GMM]" ) {
//    xmm::TrainingSet ts(true, false, 3);
//    std::vector<float> observation(3);
//    ts.addPhrase(0);
//    ts.addPhrase(1);
//    for (std::size_t i=0; i<100; i++) {
//        observation[0] = float(i)/100.;
//        observation[1] = pow(float(i)/100., 2.);
//        observation[2] = pow(float(i)/100., 3.);
//        ts.getPhrase(0)->record(observation);
//        ts.getPhrase(1)->record(observation);
//    }
//    xmm::Label label_a(static_cast<std::string>("a"));
//    xmm::Label label_b(static_cast<std::string>("b"));
//    ts.getPhrase(0)->label.set(label_a);
//    ts.getPhrase(1)->label.set(label_b);
//    xmm::GMM a(xmm::NONE, &ts);
//    a.set_nbMixtureComponents(3);
//    a.train();
//    CHECK_FALSE(a.bimodal_);
//    std::vector<float> mixtureCoeffs(3);
//    mixtureCoeffs[0] = 3.819443583488e-01;
//    mixtureCoeffs[1] = 3.171715140343e-01;
//    mixtureCoeffs[2] = 3.008841276169e-01;
//    CHECK_VECTOR_APPROX(a.models[label_a].mixtureCoeffs, mixtureCoeffs);
//    CHECK_VECTOR_APPROX(a.models[label_b].mixtureCoeffs, mixtureCoeffs);
//    std::vector<double> cov_c0(9);
//    cov_c0[0] = 1.398498568044e-02;
//    cov_c0[1] = 5.041130937517e-03;
//    cov_c0[2] = 1.792829737244e-03;
//    cov_c0[3] = 5.041130937517e-03;
//    cov_c0[4] = 3.113048333532e-03;
//    cov_c0[5] = 7.901977752380e-04;
//    cov_c0[6] = 1.792829737244e-03;
//    cov_c0[7] = 7.901977752380e-04;
//    cov_c0[8] = 1.306463534393e-03;
//    CHECK_VECTOR_APPROX(a.models[label_a].components[0].covariance, cov_c0);
//    CHECK(a.models[label_a].components[0].inverseCovariance_input_.size() == 0);
//    xmm::GMM b(a);
//    CHECK_THROWS(b.makeUnimodal());
//    CHECK_NOTHROW(b.makeBimodal(2));
//    CHECK(b.bimodal_);
//    CHECK(a.models[label_a].mixtureCoeffs == b.models[label_a].mixtureCoeffs);
//    for (std::size_t i=0; i<3; i++) {
//        CHECK(a.models[label_a].components[i].mean == b.models[label_a].components[i].mean);
//        CHECK(a.models[label_a].components[i].covariance == b.models[label_a].components[i].covariance);
//        CHECK(a.models[label_a].components[i].inverseCovariance_ == b.models[label_a].components[i].inverseCovariance_);
//        CHECK(a.models[label_a].components[i].inverseCovariance_input_.size() == 0);
//        CHECK(b.models[label_a].components[i].inverseCovariance_input_.size() == 4);
//    }
//    std::vector<double> invcov_c0_input(4);
//    invcov_c0_input[0] = 171.77395895;
//    invcov_c0_input[1] = -278.16304982;
//    invcov_c0_input[2] = -278.16304982;
//    invcov_c0_input[3] = 771.67332427;
//    CHECK_VECTOR_APPROX(b.models[label_a].components[0].inverseCovariance_input_, invcov_c0_input);
//    CHECK_VECTOR_APPROX(b.models[label_b].components[0].inverseCovariance_input_, invcov_c0_input);
//}
//
//TEST_CASE( "Convert GMM: bimodal->unimodal", "[GMM]" ) {
//    xmm::TrainingSet ts(xmm::BIMODAL, 3, 2);
//    std::vector<float> observation(3);
//    ts.addPhrase(0);
//    ts.addPhrase(1);
//    for (std::size_t i=0; i<100; i++) {
//        observation[0] = float(i)/100.;
//        observation[1] = pow(float(i)/100., 2.);
//        observation[2] = pow(float(i)/100., 3.);
//        ts.getPhrase(0)->record(observation);
//        ts.getPhrase(1)->record(observation);
//    }
//    xmm::Label label_a(static_cast<std::string>("a"));
//    xmm::Label label_b(static_cast<std::string>("b"));
//    ts.getPhrase(0)->label.set(label_a);
//    ts.getPhrase(1)->label.set(label_b);
//    xmm::GMM a(xmm::BIMODAL, &ts);
//    a.set_nbMixtureComponents(3);
//    a.train();
//    CHECK(a.bimodal_);
//    std::vector<float> mixtureCoeffs(3);
//    mixtureCoeffs[0] = 3.819443583488e-01;
//    mixtureCoeffs[1] = 3.171715140343e-01;
//    mixtureCoeffs[2] = 3.008841276169e-01;
//    CHECK_VECTOR_APPROX(a.models[label_a].mixtureCoeffs, mixtureCoeffs);
//    CHECK_VECTOR_APPROX(a.models[label_b].mixtureCoeffs, mixtureCoeffs);
//    std::vector<double> cov_c0(9);
//    cov_c0[0] = 1.398498568044e-02;
//    cov_c0[1] = 5.041130937517e-03;
//    cov_c0[2] = 1.792829737244e-03;
//    cov_c0[3] = 5.041130937517e-03;
//    cov_c0[4] = 3.113048333532e-03;
//    cov_c0[5] = 7.901977752380e-04;
//    cov_c0[6] = 1.792829737244e-03;
//    cov_c0[7] = 7.901977752380e-04;
//    cov_c0[8] = 1.306463534393e-03;
//    CHECK_VECTOR_APPROX(a.models[label_a].components[0].covariance, cov_c0);
//    xmm::GMM b(a);
//    CHECK_THROWS(b.makeBimodal(2));
//    CHECK_NOTHROW(b.makeUnimodal());
//    CHECK_FALSE(b.bimodal_);
//    CHECK(a.models[label_a].mixtureCoeffs == b.models[label_a].mixtureCoeffs);
//    for (std::size_t i=0; i<3; i++) {
//        CHECK(a.models[label_a].components[i].mean == b.models[label_a].components[i].mean);
//        CHECK(a.models[label_a].components[i].covariance == b.models[label_a].components[i].covariance);
//        CHECK(a.models[label_a].components[i].inverseCovariance_ == b.models[label_a].components[i].inverseCovariance_);
//        CHECK(b.models[label_a].components[i].inverseCovariance_input_.size() == 0);
//        CHECK(a.models[label_a].components[i].inverseCovariance_input_.size() == 4);
//    }
//    std::vector<double> invcov_c0_input(4);
//    invcov_c0_input[0] = 171.77395895;
//    invcov_c0_input[1] = -278.16304982;
//    invcov_c0_input[2] = -278.16304982;
//    invcov_c0_input[3] = 771.67332427;
//    CHECK_VECTOR_APPROX(a.models[label_a].components[0].inverseCovariance_input_, invcov_c0_input);
//    CHECK_VECTOR_APPROX(a.models[label_b].components[0].inverseCovariance_input_, invcov_c0_input);
//}
//
//TEST_CASE( "Convert HMM: unimodal->bimodal", "[HMM]" ) {
//    xmm::TrainingSet ts(true, false, 3);
//    std::vector<float> observation(3);
//    ts.addPhrase(0);
//    for (std::size_t i=0; i<100; i++) {
//        observation[0] = float(i)/100.;
//        observation[1] = pow(float(i)/100., 2.);
//        observation[2] = pow(float(i)/100., 3.);
//        ts.getPhrase(0)->record(observation);
//    }
//    xmm::HMM a(xmm::NONE);
//    a.set_nbStates(3);
//    a.train(&ts);
//    std::vector<float> transition(6);
//    transition[0] = 9.710567593575e-01;
//    transition[1] = 2.894317358732e-02;
//    transition[2] = 9.693398475647e-01;
//    transition[3] = 3.066014684737e-02;
//    transition[4] = 1;
//    transition[5] = 0;
//    CHECK_VECTOR_APPROX(a.transition, transition);
//    std::vector<double> covariance_state0(9);
//    covariance_state0[0] = 1.105028519640e-02,
//    covariance_state0[1] = 3.397482636225e-03;
//    covariance_state0[2] = 1.045622665096e-03;
//    covariance_state0[3] = 3.397482636225e-03;
//    covariance_state0[4] = 2.231733755155e-03;
//    covariance_state0[5] = 3.960464278701e-04;
//    covariance_state0[6] = 1.045622665096e-03;
//    covariance_state0[7] = 3.960464278701e-04;
//    covariance_state0[8] = 1.131260901171e-03;
//    CHECK_VECTOR_APPROX(a.states[0].components[0].covariance, covariance_state0);
//    CHECK_FALSE(a.bimodal_);
//    CHECK(a.states[0].components[0].inverseCovariance_input_.size() == 0);
//    xmm::HMM b(a);
//    b.makeBimodal(2);
//    CHECK(b.bimodal_);
//    CHECK(a.prior == b.prior);
//    CHECK(a.transition == b.transition);
//    for (std::size_t i=0; i<a.get_nbStates(); i++) {
//        CHECK(a.states[i].components[0].mean == b.states[i].components[0].mean);
//        CHECK(a.states[i].components[0].covariance == b.states[i].components[0].covariance);
//        CHECK(a.states[i].components[0].inverseCovariance_ == b.states[i].components[0].inverseCovariance_);
//        CHECK(a.states[i].components[0].inverseCovariance_input_.size() == 0);
//        CHECK(b.states[i].components[0].inverseCovariance_input_.size() == 4);
//    }
//    std::vector<double> invcov_state0_input(4);
//    invcov_state0_input[0] = 170.12232422;
//    invcov_state0_input[1] = -258.98593022;
//    invcov_state0_input[2] = -258.98593022;
//    invcov_state0_input[3] = 842.34967393;
//    CHECK_VECTOR_APPROX(b.states[0].components[0].inverseCovariance_input_, invcov_state0_input);
//}
//
//TEST_CASE( "Convert HMM: bimodal->unimodal", "[HMM]" ) {
//    xmm::TrainingSet ts(xmm::BIMODAL, 3, 2);
//    std::vector<float> observation(3);
//    ts.addPhrase(0);
//    for (std::size_t i=0; i<100; i++) {
//        observation[0] = float(i)/100.;
//        observation[1] = pow(float(i)/100., 2.);
//        observation[2] = pow(float(i)/100., 3.);
//        ts.getPhrase(0)->record(observation);
//    }
//    xmm::HMM a(xmm::BIMODAL);
//    a.set_nbStates(3);
//    a.train(&ts);
//    std::vector<float> transition(6);
//    transition[0] = 9.710567593575e-01;
//    transition[1] = 2.894317358732e-02;
//    transition[2] = 9.693398475647e-01;
//    transition[3] = 3.066014684737e-02;
//    transition[4] = 1;
//    transition[5] = 0;
//    CHECK_VECTOR_APPROX(a.transition, transition);
//    std::vector<double> covariance_state0(9);
//    covariance_state0[0] = 1.105028519640e-02,
//    covariance_state0[1] = 3.397482636225e-03;
//    covariance_state0[2] = 1.045622665096e-03;
//    covariance_state0[3] = 3.397482636225e-03;
//    covariance_state0[4] = 2.231733755155e-03;
//    covariance_state0[5] = 3.960464278701e-04;
//    covariance_state0[6] = 1.045622665096e-03;
//    covariance_state0[7] = 3.960464278701e-04;
//    covariance_state0[8] = 1.131260901171e-03;
//    CHECK_VECTOR_APPROX(a.states[0].components[0].covariance, covariance_state0);
//    std::vector<double> invcov_state0_input(4);
//    invcov_state0_input[0] = 170.12232422;
//    invcov_state0_input[1] = -258.98593022;
//    invcov_state0_input[2] = -258.98593022;
//    invcov_state0_input[3] = 842.34967393;
//    CHECK_VECTOR_APPROX(a.states[0].components[0].inverseCovariance_input_, invcov_state0_input);
//    CHECK(a.bimodal_);
//    CHECK(a.states[0].components[0].inverseCovariance_input_.size() == 4);
//    xmm::HMM b(a);
//    CHECK_THROWS(b.makeBimodal(2););
//    CHECK_NOTHROW(b.makeUnimodal());
//    CHECK_FALSE(b.bimodal_);
//    CHECK(a.prior == b.prior);
//    CHECK(a.transition == b.transition);
//    for (std::size_t i=0; i<a.get_nbStates(); i++) {
//        CHECK(a.states[i].components[0].mean == b.states[i].components[0].mean);
//        CHECK(a.states[i].components[0].covariance == b.states[i].components[0].covariance);
//        CHECK(a.states[i].components[0].inverseCovariance_ == b.states[i].components[0].inverseCovariance_);
//        CHECK(a.states[i].components[0].inverseCovariance_input_.size() == 4);
//        CHECK(b.states[i].components[0].inverseCovariance_input_.size() == 0);
//    }
//}
//
//
//TEST_CASE( "Convert HierarchicalHMM: unimodal->bimodal", "[HierarchicalHMM]" ) {
//    xmm::TrainingSet ts(true, false, 3);
//    std::vector<float> observation(3);
//    ts.addPhrase(0);
//    ts.addPhrase(1);
//    for (std::size_t i=0; i<100; i++) {
//        observation[0] = float(i)/100.;
//        observation[1] = pow(float(i)/100., 2.);
//        observation[2] = pow(float(i)/100., 3.);
//        ts.getPhrase(0)->record(observation);
//        ts.getPhrase(1)->record(observation);
//    }
//    xmm::Label label_a(static_cast<std::string>("a"));
//    xmm::Label label_b(static_cast<std::string>("b"));
//    ts.getPhrase(0)->label.set(label_a);
//    ts.getPhrase(1)->label.set(label_b);
//    xmm::HierarchicalHMM a(xmm::NONE, &ts);
//    a.set_nbStates(3);
//    a.train();
//    std::vector<float> transition(6);
//    transition[0] = 9.710567593575e-01;
//    transition[1] = 2.894317358732e-02;
//    transition[2] = 9.693398475647e-01;
//    transition[3] = 3.066014684737e-02;
//    transition[4] = 1;
//    transition[5] = 0;
//    CHECK_VECTOR_APPROX(a.models[label_a].transition, transition);
//    std::vector<double> covariance_state0(9);
//    covariance_state0[0] = 1.105028519640e-02,
//    covariance_state0[1] = 3.397482636225e-03;
//    covariance_state0[2] = 1.045622665096e-03;
//    covariance_state0[3] = 3.397482636225e-03;
//    covariance_state0[4] = 2.231733755155e-03;
//    covariance_state0[5] = 3.960464278701e-04;
//    covariance_state0[6] = 1.045622665096e-03;
//    covariance_state0[7] = 3.960464278701e-04;
//    covariance_state0[8] = 1.131260901171e-03;
//    CHECK_VECTOR_APPROX(a.models[label_a].states[0].components[0].covariance, covariance_state0);
//    CHECK_FALSE(a.bimodal_);
//    CHECK(a.models[label_a].states[0].components[0].inverseCovariance_input_.size() == 0);
//    xmm::HierarchicalHMM b(a);
//    b.makeBimodal(2);
//    CHECK(b.bimodal_);
//    CHECK(a.models[label_a].prior == b.models[label_a].prior);
//    CHECK(a.models[label_a].transition == b.models[label_a].transition);
//    for (std::size_t i=0; i<a.get_nbStates(); i++) {
//        CHECK(a.models[label_a].states[i].components[0].mean == b.models[label_a].states[i].components[0].mean);
//        CHECK(a.models[label_a].states[i].components[0].covariance == b.models[label_a].states[i].components[0].covariance);
//        CHECK(a.models[label_a].states[i].components[0].inverseCovariance_ == b.models[label_a].states[i].components[0].inverseCovariance_);
//        CHECK(a.models[label_a].states[i].components[0].inverseCovariance_input_.size() == 0);
//        CHECK(b.models[label_a].states[i].components[0].inverseCovariance_input_.size() == 4);
//    }
//    std::vector<double> invcov_state0_input(4);
//    invcov_state0_input[0] = 170.12232422;
//    invcov_state0_input[1] = -258.98593022;
//    invcov_state0_input[2] = -258.98593022;
//    invcov_state0_input[3] = 842.34967393;
//    CHECK_VECTOR_APPROX(b.models[label_a].states[0].components[0].inverseCovariance_input_, invcov_state0_input);
//}
//
//TEST_CASE( "Convert HierarchicalHMM: bimodal->unimodal", "[HierarchicalHMM]" ) {
//    xmm::TrainingSet ts(xmm::BIMODAL, 3, 2);
//    std::vector<float> observation(3);
//    ts.addPhrase(0);
//    ts.addPhrase(1);
//    for (std::size_t i=0; i<100; i++) {
//        observation[0] = float(i)/100.;
//        observation[1] = pow(float(i)/100., 2.);
//        observation[2] = pow(float(i)/100., 3.);
//        ts.getPhrase(0)->record(observation);
//        ts.getPhrase(1)->record(observation);
//    }
//    xmm::Label label_a(static_cast<std::string>("a"));
//    xmm::Label label_b(static_cast<std::string>("b"));
//    ts.getPhrase(0)->label.set(label_a);
//    ts.getPhrase(1)->label.set(label_b);
//    xmm::HierarchicalHMM a(xmm::BIMODAL, &ts);
//    a.set_nbStates(3);
//    a.train();
//    std::vector<float> transition(6);
//    transition[0] = 9.710567593575e-01;
//    transition[1] = 2.894317358732e-02;
//    transition[2] = 9.693398475647e-01;
//    transition[3] = 3.066014684737e-02;
//    transition[4] = 1;
//    transition[5] = 0;
//    CHECK_VECTOR_APPROX(a.models[label_a].transition, transition);
//    std::vector<double> covariance_state0(9);
//    covariance_state0[0] = 1.105028519640e-02,
//    covariance_state0[1] = 3.397482636225e-03;
//    covariance_state0[2] = 1.045622665096e-03;
//    covariance_state0[3] = 3.397482636225e-03;
//    covariance_state0[4] = 2.231733755155e-03;
//    covariance_state0[5] = 3.960464278701e-04;
//    covariance_state0[6] = 1.045622665096e-03;
//    covariance_state0[7] = 3.960464278701e-04;
//    covariance_state0[8] = 1.131260901171e-03;
//    CHECK_VECTOR_APPROX(a.models[label_a].states[0].components[0].covariance, covariance_state0);
//    std::vector<double> invcov_state0_input(4);
//    invcov_state0_input[0] = 170.12232422;
//    invcov_state0_input[1] = -258.98593022;
//    invcov_state0_input[2] = -258.98593022;
//    invcov_state0_input[3] = 842.34967393;
//    CHECK_VECTOR_APPROX(a.models[label_a].states[0].components[0].inverseCovariance_input_, invcov_state0_input);
//    CHECK(a.bimodal_);
//    CHECK(a.models[label_a].states[0].components[0].inverseCovariance_input_.size() == 4);
//    xmm::HierarchicalHMM b(a);
//    CHECK_THROWS(b.makeBimodal(2););
//    CHECK_NOTHROW(b.makeUnimodal());
//    CHECK_FALSE(b.bimodal_);
//    CHECK(a.models[label_a].prior == b.models[label_a].prior);
//    CHECK(a.models[label_a].transition == b.models[label_a].transition);
//    CHECK(a.models[label_b].prior == b.models[label_b].prior);
//    CHECK(a.models[label_b].transition == b.models[label_b].transition);
//    for (std::size_t i=0; i<a.get_nbStates(); i++) {
//        CHECK(a.models[label_a].states[i].components[0].mean == b.models[label_a].states[i].components[0].mean);
//        CHECK(a.models[label_a].states[i].components[0].covariance == b.models[label_a].states[i].components[0].covariance);
//        CHECK(a.models[label_a].states[i].components[0].inverseCovariance_ == b.models[label_a].states[i].components[0].inverseCovariance_);
//        CHECK(a.models[label_a].states[i].components[0].inverseCovariance_input_.size() == 4);
//        CHECK(b.models[label_a].states[i].components[0].inverseCovariance_input_.size() == 0);
//    }
//}
//
