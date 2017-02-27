/*
 * xmmTestsModelConversion.cpp
 *
 * Test suite for the conversion (unimodal <-> bimodal)
 *
 * Contact:
 * - Jules Francoise <jules.francoise@ircam.fr>
 *
 * This code has been initially authored by Jules Francoise
 * <http://julesfrancoise.com> during his PhD thesis, supervised by Frederic
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

TEST_CASE( "Convert Gaussian Distribution: unimodal->bimodal", "[GaussianDistribution]" ) {
    xmm::GaussianDistribution a(false, 3, 0);
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
    CHECK_THROWS(a.getUnimodal());
    xmm::GaussianDistribution b;
    CHECK_NOTHROW(b = a.getBimodal(2));
    CHECK_FALSE(a.bimodal_);
    CHECK(b.bimodal_);
    CHECK(a.dimension.get() == 3);
    CHECK(a.dimension_input.get() == 0);
    CHECK(b.dimension.get() == 3);
    CHECK(b.dimension_input.get() == 2);
    CHECK(a.mean == b.mean);
    CHECK(a.covariance == b.covariance);
    CHECK(a.inverse_covariance_ == b.inverse_covariance_);
    std::vector<double> invcov(9);
    invcov[0] = 0.79037801;
    invcov[1] = 0.06872852;
    invcov[2] = -0.13745704;
    invcov[3] = 0.06872852;
    invcov[4] = 0.93765341;
    invcov[5] = -0.4467354;
    invcov[6] = -0.13745704;
    invcov[7] = -0.4467354;
    invcov[8] = 0.89347079;
    CHECK_VECTOR_APPROX(b.inverse_covariance_, invcov);
    std::vector<double> invcov_input(4);
    invcov_input[0] = 0.76923077;
    invcov_input[1] = 0.;
    invcov_input[2] = 0.;
    invcov_input[3] = 0.71428571;
    CHECK(a.inverse_covariance_input_.empty());
    CHECK_VECTOR_APPROX(b.inverse_covariance_input_, invcov_input);
}

TEST_CASE( "Convert Gaussian Distribution: bimodal->unimodal", "[GaussianDistribution]" ) {
    xmm::GaussianDistribution a(true, 3, 2);
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
    CHECK_THROWS(a.getBimodal(2));
    xmm::GaussianDistribution b;
    CHECK_NOTHROW(b = a.getUnimodal());
    CHECK(a.bimodal_);
    CHECK_FALSE(b.bimodal_);
    CHECK(a.dimension.get() == 3);
    CHECK(a.dimension_input.get() == 2);
    CHECK(b.dimension.get() == 3);
    CHECK(b.dimension_input.get() == 0);
    CHECK(a.mean == b.mean);
    CHECK(a.covariance == b.covariance);
    CHECK(a.inverse_covariance_ == b.inverse_covariance_);
    std::vector<double> invcov(9);
    invcov[0] = 0.79037801;
    invcov[1] = 0.06872852;
    invcov[2] = -0.13745704;
    invcov[3] = 0.06872852;
    invcov[4] = 0.93765341;
    invcov[5] = -0.4467354;
    invcov[6] = -0.13745704;
    invcov[7] = -0.4467354;
    invcov[8] = 0.89347079;
    CHECK_VECTOR_APPROX(b.inverse_covariance_, invcov);
    std::vector<double> invcov_input(4);
    invcov_input[0] = 0.76923077;
    invcov_input[1] = 0.;
    invcov_input[2] = 0.;
    invcov_input[3] = 0.71428571;
    CHECK(b.inverse_covariance_input_.empty());
    CHECK_VECTOR_APPROX(a.inverse_covariance_input_, invcov_input);

}

TEST_CASE( "Convert GMM: unimodal->bimodal (2 classes)", "[GMM]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory, xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    std::vector<float> observation(3);
    ts.addPhrase(0, "a");
    ts.addPhrase(1, "b");
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    xmm::GMM a(false);
    a.configuration.gaussians.set(3);
    a.train(&ts);
    CHECK_FALSE(a.shared_parameters->bimodal.get());
    CHECK(a.models["a"].components[0].inverse_covariance_input_.size() == 0);
    xmm::GMM b;
    CHECK_THROWS(a.getUnimodal());
    CHECK_NOTHROW(b = a.getBimodal(2));
    CHECK(b.shared_parameters->bimodal.get());
    CHECK(a.models["a"].mixture_coeffs == b.models["a"].mixture_coeffs);
    for (unsigned int i=0; i<3; i++) {
        CHECK(a.models["a"].components[i].mean == b.models["a"].components[i].mean);
        CHECK(a.models["a"].components[i].covariance == b.models["a"].components[i].covariance);
        CHECK(a.models["a"].components[i].inverse_covariance_ == b.models["a"].components[i].inverse_covariance_);
        CHECK(a.models["a"].components[i].inverse_covariance_input_.size() == 0);
        CHECK(b.models["a"].components[i].inverse_covariance_input_.size() == 4);
    }
}

TEST_CASE( "Convert GMM: bimodal->unimodal (2 classes)", "[GMM]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory, xmm::Multimodality::Bimodal);
    ts.dimension.set(3);
    ts.dimension_input.set(2);
    std::vector<float> observation(3);
    ts.addPhrase(0, "a");
    ts.addPhrase(1, "b");
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    xmm::GMM a(true);
    a.configuration.gaussians.set(3);
    a.train(&ts);
    xmm::GMM b;
    CHECK_THROWS(a.getBimodal(2));
    CHECK_NOTHROW(b = a.getUnimodal());
    CHECK_FALSE(b.shared_parameters->bimodal.get());
    CHECK(a.models["a"].mixture_coeffs == b.models["a"].mixture_coeffs);
    for (unsigned int i=0; i<3; i++) {
        CHECK(a.models["a"].components[i].mean == b.models["a"].components[i].mean);
        CHECK(a.models["a"].components[i].covariance == b.models["a"].components[i].covariance);
        CHECK(a.models["a"].components[i].inverse_covariance_ == b.models["a"].components[i].inverse_covariance_);
        CHECK(b.models["a"].components[i].inverse_covariance_input_.size() == 0);
        CHECK(a.models["a"].components[i].inverse_covariance_input_.size() == 4);
    }
}

TEST_CASE( "Convert HierarchicalHMM: unimodal->bimodal", "[HierarchicalHMM]") {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory, xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    std::vector<float> observation(3);
    ts.addPhrase(0, "a");
    ts.addPhrase(1, "b");
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    xmm::HierarchicalHMM a(false);
    a.configuration.states.set(3);
    a.train(&ts);
    CHECK_FALSE(a.shared_parameters->bimodal.get());
    CHECK(a.models["a"].states[0].components[0].inverse_covariance_input_.size() == 0);
    xmm::HierarchicalHMM b = a.getBimodal(2);
    CHECK(b.shared_parameters->bimodal.get());
    CHECK(a.models["a"].prior == b.models["a"].prior);
    CHECK(a.models["a"].transition == b.models["a"].transition);
    for (unsigned int i=0; i<a.configuration.states.get(); i++) {
        CHECK(a.models["a"].states[i].components[0].mean == b.models["a"].states[i].components[0].mean);
        CHECK(a.models["a"].states[i].components[0].covariance == b.models["a"].states[i].components[0].covariance);
        CHECK(a.models["a"].states[i].components[0].inverse_covariance_ == b.models["a"].states[i].components[0].inverse_covariance_);
        CHECK(a.models["a"].states[i].components[0].inverse_covariance_input_.size() == 0);
        CHECK(b.models["a"].states[i].components[0].inverse_covariance_input_.size() == 4);
    }
}

TEST_CASE( "Convert HierarchicalHMM: bimodal->unimodal", "[HierarchicalHMM]") {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory, xmm::Multimodality::Bimodal);
    ts.dimension.set(3);
    ts.dimension_input.set(2);
    std::vector<float> observation(3);
    ts.addPhrase(0, "a");
    ts.addPhrase(1, "b");
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    xmm::HierarchicalHMM a(true);
    a.configuration.states.set(3);
    a.train(&ts);
    CHECK(a.shared_parameters->bimodal.get());
    CHECK(a.models["a"].states[0].components[0].inverse_covariance_input_.size() == 4);
    xmm::HierarchicalHMM b;
    CHECK_THROWS(a.getBimodal(2););
    CHECK_NOTHROW(b = a.getUnimodal());
    CHECK_FALSE(b.shared_parameters->bimodal.get());
    CHECK(a.models["a"].prior == b.models["a"].prior);
    CHECK(a.models["a"].transition == b.models["a"].transition);
    CHECK(a.models["b"].prior == b.models["b"].prior);
    CHECK(a.models["b"].transition == b.models["b"].transition);
    for (unsigned int i=0; i<a.configuration.states.get(); i++) {
        CHECK(a.models["a"].states[i].components[0].mean == b.models["a"].states[i].components[0].mean);
        CHECK(a.models["a"].states[i].components[0].covariance == b.models["a"].states[i].components[0].covariance);
        CHECK(a.models["a"].states[i].components[0].inverse_covariance_ == b.models["a"].states[i].components[0].inverse_covariance_);
        CHECK(a.models["a"].states[i].components[0].inverse_covariance_input_.size() == 4);
        CHECK(b.models["a"].states[i].components[0].inverse_covariance_input_.size() == 0);
    }
}

