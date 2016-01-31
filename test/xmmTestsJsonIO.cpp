/*
 * xmmTestsJsonIO.cpp
 *
 * Test suite for Json I/O
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
#include <ctime>

TEST_CASE( "Phrase: Json IO", "[JSON I/O]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    std::vector<float> observation(3);
    std::string label_a(static_cast<std::string>("a"));
    std::string label_b(static_cast<std::string>("b"));
    ts.addPhrase(0, label_a);
    ts.addPhrase(1, label_b);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.getPhrase(0)->record(observation);
        observation[0] -= 1.;
        observation[1] -= 1.;
        observation[2] -= 1.;
        ts.getPhrase(1)->record(observation);
    }
    CHECK(ts.getPhrase(0)->toJson() != ts.getPhrase(1)->toJson());
    //    std::cout << ts.getPhrase(0)->toJson() << std::endl;
    //    std::cout << ts.getPhrase(1)->toJson() << std::endl;
    ts.getPhrase(1)->fromJson(ts.getPhrase(0)->toJson());
    CHECK(ts.getPhrase(0)->toJson() == ts.getPhrase(1)->toJson());
    //    std::cout << ts.getPhrase(1)->toJson() << std::endl;
    
    xmm::TrainingSet ts2(xmm::MemoryMode::OwnMemory,
                         xmm::Multimodality::Unimodal);
    ts2.dimension.set(2);
    std::string label_c(static_cast<std::string>("c"));
    ts2.addPhrase(0, label_c);
    CHECK_NOTHROW(ts2.getPhrase(0)->fromJson(ts.getPhrase(0)->toJson()));
    CHECK(ts2.getPhrase(0)->toJson() == ts.getPhrase(0)->toJson());
    //    std::cout << ts2.getPhrase(0)->toJson() << std::endl;
    
    xmm::TrainingSet ts3(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Bimodal);
    ts3.dimension.set(3);
    ts3.dimension_input.set(2);
    std::vector<float> observation_input(2);
    std::vector<float> observation_output(1);
    ts3.addPhrase(0, label_a);
    for (unsigned int i=0; i<100; i++) {
        observation_input[0] = float(i)/100.;
        observation_input[1] = pow(float(i)/100., 2.);
        observation_output[0] = pow(float(i)/100., 3.);
        ts3.getPhrase(0)->record_input(observation_input);
        ts3.getPhrase(0)->record_output(observation_output);
    }
    ts3.addPhrase(1243, label_c);
    CHECK_NOTHROW(ts3.getPhrase(1243)->fromJson(ts3.getPhrase(0)->toJson()));
    for (int i=0; i<ts3.getPhrase(0)->size(); i++) {
        CHECK(ts3.getPhrase(0)->getValue(i, 0) == ts3.getPhrase(1243)->getValue(i, 0));
    }
    CHECK_NOTHROW(ts3.getPhrase(1243)->fromJson(ts.getPhrase(0)->toJson()));
}

TEST_CASE( "Training Set: Json IO", "[JSON I/O]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    std::vector<float> observation(3);
    std::string label_a(static_cast<std::string>("a"));
    std::string label_b(static_cast<std::string>("b"));
    ts.addPhrase(12, label_a);
    ts.addPhrase(18, label_b);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.getPhrase(12)->record(observation);
        observation[0] -= 1.;
        observation[1] -= 1.;
        observation[2] -= 1.;
        ts.getPhrase(18)->record(observation);
    }
    // std::cout << ts.toJson() << std::endl;

    xmm::TrainingSet ts2(xmm::MemoryMode::OwnMemory,
                         xmm::Multimodality::Bimodal);
    ts2.dimension.set(2);
    std::string label_c(static_cast<std::string>("c"));
    CHECK_NOTHROW(ts2.fromJson(ts.toJson()));
    CHECK(ts.toJson() == ts2.toJson());
}

TEST_CASE( "GaussianDistribution: Json IO", "[JSON I/O]" ) {
    xmm::GaussianDistribution a;
    a.dimension.set(3);
    a.mean = {1, 2, 3};
    a.covariance = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    
    // std::cout << a.toJson() << std::endl;
    xmm::GaussianDistribution b(a.toJson());
    CHECK(a.toJson() == b.toJson());
    
    xmm::GaussianDistribution c;
    CHECK_NOTHROW(c.fromJson(a.toJson()));
}

TEST_CASE( "GMM: Json IO", "[JSON I/O]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    std::vector<float> observation(3);
    std::string label_a(static_cast<std::string>("a"));
    std::string label_b(static_cast<std::string>("b"));
    ts.addPhrase(0, label_a);
    ts.addPhrase(1, label_b);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    xmm::GMM a;
    a.configuration.gaussians.set(3);
    a.configuration.absolute_regularization.set(1.);
    a.configuration.relative_regularization.set(1.);
    a.shared_parameters->em_algorithm_percent_chg.set(0.);
    a.train(&ts);
    
    xmm::GMM b(a.toJson());
    CHECK(b.toJson() == a.toJson());
    
    xmm::GMM c;
    c.fromJson(a.toJson());
    CHECK(c.toJson() == a.toJson());
}

TEST_CASE( "HierarchicalHMM: Json IO", "[JSON I/O]" ) {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    ts.column_names.set({"x", "y", "z"});
    std::vector<float> observation(3);
    std::string label_a(static_cast<std::string>("a"));
    std::string label_b(static_cast<std::string>("b"));
    ts.addPhrase(0, label_a);
    ts.addPhrase(1, label_b);
    for (unsigned int i=0; i<100; i++) {
        observation[0] = float(i)/100.;
        observation[1] = pow(float(i)/100., 2.);
        observation[2] = pow(float(i)/100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    xmm::HierarchicalHMM a;
    a.configuration.states.set(3);
    a.configuration.absolute_regularization.set(1.);
    a.configuration.relative_regularization.set(1.);
    a.configuration[label_b].states.set(12);
    a.shared_parameters->em_algorithm_percent_chg.set(0.);
    a.train(&ts);
    
    xmm::HierarchicalHMM b(a.toJson());
    CHECK(b.toJson() == a.toJson());
    
    xmm::HierarchicalHMM c;
    c.fromJson(a.toJson());
    CHECK(c.toJson() == a.toJson());
}
