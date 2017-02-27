/*
 * xmmTestsModelExtraction.cpp
 *
 * Test suite for the extraction (submodel, reverse model)
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

TEST_CASE( "Extract Gaussian Distribution: arbitrary dimensions", "[GaussianDistribution]" ) {
    xmm::GaussianDistribution a(false,
                           3,
                           0);
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
    std::vector<unsigned int> columns(2);
    columns[0] = 2;
    columns[1] = 0;
    xmm::GaussianDistribution b = a.extractSubmodel(columns);
    CHECK_FALSE(b.bimodal_);
    CHECK(b.dimension.get() == 2);
    std::vector<double> new_covariance(4);
    new_covariance[0] = 1.5;
    new_covariance[1] = 0.2;
    new_covariance[2] = 0.2;
    new_covariance[3] = 1.3;
    CHECK_VECTOR_APPROX(b.covariance, new_covariance);
    CHECK(b.mean[0] == Approx(a.mean[2]));
    CHECK(b.mean[1] == Approx(a.mean[0]));
}


TEST_CASE( "Extract Gaussian Distribution: input & output", "[GaussianDistribution]" ) {
    xmm::GaussianDistribution a(true,
                           3,
                           2);
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
    xmm::GaussianDistribution b = a.extractSubmodel_input();
    CHECK_FALSE(b.bimodal_);
    CHECK(b.dimension.get() == 2);
    std::vector<double> new_covariance(4);
    new_covariance[0] = 1.3;
    new_covariance[1] = 0.;
    new_covariance[2] = 0.;
    new_covariance[3] = 1.4;
    CHECK_VECTOR_APPROX(b.covariance, new_covariance);
    CHECK(b.mean[0] == Approx(a.mean[0]));
    CHECK(b.mean[1] == Approx(a.mean[1]));
    xmm::GaussianDistribution c = a.extractSubmodel_output();
    CHECK_FALSE(c.bimodal_);
    CHECK(c.dimension.get() == 1);
    new_covariance.resize(1);
    new_covariance[0] = 1.5;
    CHECK_VECTOR_APPROX(c.covariance, new_covariance);
    CHECK(c.mean[0] == Approx(a.mean[2]));
    CHECK(c.mean.size() == 1);
}

TEST_CASE( "Extract Gaussian Distribution: inverse model", "[GaussianDistribution]" ) {
    xmm::GaussianDistribution a(true,
                           3,
                           2);
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
    xmm::GaussianDistribution b = a.extract_inverse_model();
    CHECK(b.bimodal_);
    CHECK(b.dimension.get() == 3);
    std::vector<double> new_covariance(9);
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
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
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
    std::vector<unsigned int> columns(2);
    columns[0] = 2;
    columns[1] = 0;
    xmm::GMM b = a.extractSubmodel(columns);
    CHECK_FALSE(b.shared_parameters->bimodal.get());
    CHECK(b.models["a"].shared_parameters->dimension.get() == 2);
    CHECK_VECTOR_APPROX(b.models["a"].mixture_coeffs, a.models["a"].mixture_coeffs);
    CHECK(b.models["a"].components[0].mean[0] == Approx(a.models["a"].components[0].mean[2]));
    CHECK(b.models["a"].components[0].mean[1] == Approx(a.models["a"].components[0].mean[0]));
    CHECK(b.models["a"].components[0].covariance[0] == Approx(a.models["a"].components[0].covariance[8]));
    CHECK(b.models["a"].components[0].covariance[3] == Approx(a.models["a"].components[0].covariance[0]));
    CHECK(b.models["a"].components[0].covariance[1] == Approx(a.models["a"].components[0].covariance[2]));
    CHECK_VECTOR_APPROX(b.models["b"].mixture_coeffs, a.models["b"].mixture_coeffs);
    CHECK(b.models["b"].components[0].mean[0] == Approx(a.models["b"].components[0].mean[2]));
    CHECK(b.models["b"].components[0].mean[1] == Approx(a.models["b"].components[0].mean[0]));
}


 TEST_CASE( "Extract GMM: input & output", "[GMM]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Bimodal);
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
    xmm::GMM b = a.extractSubmodel_input();
    CHECK_FALSE(b.shared_parameters->bimodal.get());
    CHECK(b.models["a"].shared_parameters->dimension.get() == 2);
    CHECK(b.models["a"].components[0].mean[0] == Approx(a.models["a"].components[0].mean[0]));
    CHECK(b.models["a"].components[0].mean[1] == Approx(a.models["a"].components[0].mean[1]));
    xmm::GMM c = a.extractSubmodel_output();
    CHECK_FALSE(c.shared_parameters->bimodal.get());
    CHECK(c.models["a"].shared_parameters->dimension.get() == 1);
    CHECK(c.models["a"].components[0].mean[0] == Approx(a.models["a"].components[0].mean[2]));
    CHECK(c.models["a"].components[0].mean.size() == 1);

}

 TEST_CASE( "Extract GMM: inverse model", "[GMM]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Bimodal);
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
    xmm::GMM b = a.extract_inverse_model();
    CHECK(b.shared_parameters->bimodal.get());
    CHECK(b.models["a"].shared_parameters->dimension.get() == 3);
    CHECK(b.models["a"].components[0].mean[0] == Approx(a.models["a"].components[0].mean[2]));
    CHECK(b.models["a"].components[0].mean[1] == Approx(a.models["a"].components[0].mean[0]));
    CHECK(b.models["a"].components[0].mean[2] == Approx(a.models["a"].components[0].mean[1]));
    CHECK(b.models["b"].components[0].mean[0] == Approx(a.models["b"].components[0].mean[2]));
    CHECK(b.models["b"].components[0].mean[1] == Approx(a.models["b"].components[0].mean[0]));
    CHECK(b.models["b"].components[0].mean[2] == Approx(a.models["b"].components[0].mean[1]));
}

TEST_CASE( "Extract HierarchicalHMM: arbitrary dimensions", "[HierarchicalHMM]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
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
    xmm::HierarchicalHMM a;
    a.configuration.states.set(3);
    a.train(&ts);
    CHECK_FALSE(a.shared_parameters->bimodal.get());
    CHECK(a.models["a"].states[0].components[0].inverse_covariance_input_.size() == 0);
    std::vector<unsigned int> columns(2);
    columns[0] = 2;
    columns[1] = 0;
    xmm::HierarchicalHMM b = a.extractSubmodel(columns);
    CHECK_FALSE(b.shared_parameters->bimodal.get());
    CHECK(b.shared_parameters->dimension.get() == 2);
    CHECK(b.models["a"].states[0].components[0].dimension.get() == 2);
    CHECK_VECTOR_APPROX(b.models["a"].states[0].mixture_coeffs, a.models["a"].states[0].mixture_coeffs);
    CHECK(b.models["a"].states[0].components[0].mean[0] == Approx(a.models["a"].states[0].components[0].mean[2]));
    CHECK(b.models["a"].states[0].components[0].mean[1] == Approx(a.models["a"].states[0].components[0].mean[0]));
}


TEST_CASE( "Extract HierarchicalHMM: input & output", "[HierarchicalHMM]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Bimodal);
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
    xmm::HierarchicalHMM b = a.extractSubmodel_input();
    CHECK_FALSE(b.shared_parameters->bimodal.get());
    CHECK(b.shared_parameters->dimension.get() == 2);
    CHECK(b.models["a"].states[0].components[0].dimension.get() == 2);
    CHECK_VECTOR_APPROX(b.models["a"].states[0].mixture_coeffs, a.models["a"].states[0].mixture_coeffs);
    CHECK(b.models["a"].states[0].components[0].mean[0] == Approx(a.models["a"].states[0].components[0].mean[0]));
    CHECK(b.models["a"].states[0].components[0].mean[1] == Approx(a.models["a"].states[0].components[0].mean[1]));
    xmm::HierarchicalHMM c = a.extractSubmodel_output();
    CHECK_FALSE(c.shared_parameters->bimodal.get());
    CHECK(c.shared_parameters->dimension.get() == 1);
    CHECK(c.models["a"].states[0].components[0].dimension.get() == 1);
    CHECK_VECTOR_APPROX(c.models["a"].states[0].mixture_coeffs, a.models["a"].states[0].mixture_coeffs);
    CHECK(c.models["a"].states[0].components[0].mean[0] == Approx(a.models["a"].states[0].components[0].mean[2]));
}

TEST_CASE( "Extract HierarchicalHMM: inverse model", "[HierarchicalHMM]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Bimodal);
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
    CHECK(a.models["a"].shared_parameters->dimension_input.get() == 2);
    xmm::HierarchicalHMM b = a.extract_inverse_model();
    CHECK(b.shared_parameters->bimodal.get());
    CHECK(b.models["a"].shared_parameters->dimension.get() == 3);
    CHECK(b.models["a"].shared_parameters->dimension_input.get() == 1);
    CHECK(b.models["a"].states[0].components[0].mean[0] == Approx(a.models["a"].states[0].components[0].mean[2]));
    CHECK(b.models["a"].states[0].components[0].mean[1] == Approx(a.models["a"].states[0].components[0].mean[0]));
    CHECK(b.models["a"].states[0].components[0].mean[2] == Approx(a.models["a"].states[0].components[0].mean[1]));
    b.reset();
    observation.resize(1);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = pow(float(i)/100., 3.);
        b.filter(observation);
    }
}

