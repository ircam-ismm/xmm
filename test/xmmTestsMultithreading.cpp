/*
 * xmmTestsMultithreading.cpp
 *
 * Test suite for Multithreading modes of the training algorithms
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
#include <ctime>
#include <iostream>

class MyTrainingListener {
  public:
    void onEvent(xmm::TrainingEvent const &e) {
        m.lock();
        std::string status;
        if (e.status == xmm::TrainingEvent::Status::Run)
            status = "Run";
        else if (e.status == xmm::TrainingEvent::Status::Done)
            status = "Done";
        else if (e.status == xmm::TrainingEvent::Status::Cancel)
            status = "Cancel";
        else if (e.status == xmm::TrainingEvent::Status::Error)
            status = "Error";
        else if (e.status == xmm::TrainingEvent::Status::Alldone)
            status = "Alldone";
        if (e.status != xmm::TrainingEvent::Status::Alldone) {
            //            std::cout << "Class " + e.label + ": " + status + "
            //            (iteration: " + std::to_string(e.iterations) + ",
            //            progression: " + std::to_string(e.progression) + ",
            //            log_likelihood: " + std::to_string(e.log_likelihood) +
            //            ")" << std::endl;
        } else {
            //            std::cout << "All models trained." << std::endl;
        }
        m.unlock();
    }

    std::mutex m;
};

TEST_CASE("Training with MultithreadingMode::Sequential", "[GMM]") {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    std::vector<float> observation(3);
    std::string label_a(static_cast<std::string>("a"));
    std::string label_b(static_cast<std::string>("b"));
    ts.addPhrase(0, label_a);
    ts.addPhrase(1, label_b);
    for (unsigned int i = 0; i < 100; i++) {
        observation[0] = float(i) / 100.;
        observation[1] = pow(float(i) / 100., 2.);
        observation[2] = pow(float(i) / 100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    xmm::GMM a;
    MyTrainingListener *listener = new MyTrainingListener();
    a.training_events.addListener(listener, &MyTrainingListener::onEvent);
    a.configuration.multithreading = xmm::MultithreadingMode::Sequential;
    a.configuration.gaussians.set(3);
    CHECK_NOTHROW(a.train(&ts));
    std::vector<float> mixtureCoeffs(3);
    mixtureCoeffs[0] = 0.410568;
    mixtureCoeffs[1] = 0.309933;
    mixtureCoeffs[2] = 0.2795;
    CHECK_VECTOR_APPROX(a.models[label_a].mixture_coeffs, mixtureCoeffs);
    std::vector<double> cov_c0(9);
    cov_c0[0] = 0.018729;
    cov_c0[1] = 0.00683308;
    cov_c0[2] = 0.00271096;
    cov_c0[3] = 0.00683308;
    cov_c0[4] = 0.00614827;
    cov_c0[5] = 0.00133768;
    cov_c0[6] = 0.00271096;
    cov_c0[7] = 0.00133768;
    cov_c0[8] = 0.00337321;
    CHECK_VECTOR_APPROX(a.models[label_a].components[0].covariance, cov_c0);
}

TEST_CASE("Training with MultithreadingMode::MultithreadingMode", "[GMM]") {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    std::vector<float> observation(3);
    ts.addPhrase(0);
    ts.addPhrase(1);
    for (unsigned int i = 0; i < 100; i++) {
        observation[0] = float(i) / 100.;
        observation[1] = pow(float(i) / 100., 2.);
        observation[2] = pow(float(i) / 100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    std::string label_a(static_cast<std::string>("a"));
    std::string label_b(static_cast<std::string>("b"));
    ts.getPhrase(0)->label.set(label_a);
    ts.getPhrase(1)->label.set(label_b);
    xmm::GMM a;
    MyTrainingListener *listener = new MyTrainingListener();
    a.training_events.addListener(listener, &MyTrainingListener::onEvent);
    a.configuration.multithreading = xmm::MultithreadingMode::Parallel;
    a.configuration.gaussians.set(3);
    a.train(&ts);
    std::vector<float> mixtureCoeffs(3);
    mixtureCoeffs[0] = 0.410568;
    mixtureCoeffs[1] = 0.309933;
    mixtureCoeffs[2] = 0.2795;
    CHECK_VECTOR_APPROX(a.models[label_a].mixture_coeffs, mixtureCoeffs);
    std::vector<double> cov_c0(9);
    cov_c0[0] = 0.018729;
    cov_c0[1] = 0.00683308;
    cov_c0[2] = 0.00271096;
    cov_c0[3] = 0.00683308;
    cov_c0[4] = 0.00614827;
    cov_c0[5] = 0.00133768;
    cov_c0[6] = 0.00271096;
    cov_c0[7] = 0.00133768;
    cov_c0[8] = 0.00337321;
    CHECK_VECTOR_APPROX(a.models[label_a].components[0].covariance, cov_c0);
}

class BackgroundListener {
  public:
    bool trained() { return trained_; }

    void onTrainingEvent(xmm::TrainingEvent &e) {
        trained_ = (e.status == xmm::TrainingEvent::Status::Alldone);
    }

  protected:
    bool trained_ = false;
};

TEST_CASE("Training with MultithreadingMode::Background", "[GMM]") {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    std::vector<float> observation(3);
    ts.addPhrase(0);
    ts.addPhrase(1);
    for (unsigned int i = 0; i < 100; i++) {
        observation[0] = float(i) / 100.;
        observation[1] = pow(float(i) / 100., 2.);
        observation[2] = pow(float(i) / 100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    std::string label_a(static_cast<std::string>("a"));
    std::string label_b(static_cast<std::string>("b"));
    ts.getPhrase(0)->label.set(label_a);
    ts.getPhrase(1)->label.set(label_b);
    xmm::GMM a;
    a.configuration.multithreading = xmm::MultithreadingMode::Background;
    a.configuration.gaussians.set(3);
    BackgroundListener listener;
    a.training_events.addListener(&listener,
                                  &BackgroundListener::onTrainingEvent);
    a.train(&ts);
    // std::cout << "training";
    while (!listener.trained()) {
        std::cout << "";
    }
    // std::cout << std::endl;
    std::vector<float> mixtureCoeffs(3);
    mixtureCoeffs[0] = 0.410568;
    mixtureCoeffs[1] = 0.309933;
    mixtureCoeffs[2] = 0.2795;
    CHECK_VECTOR_APPROX(a.models[label_a].mixture_coeffs, mixtureCoeffs);
    std::vector<double> cov_c0(9);
    cov_c0[0] = 0.018729;
    cov_c0[1] = 0.00683308;
    cov_c0[2] = 0.00271096;
    cov_c0[3] = 0.00683308;
    cov_c0[4] = 0.00614827;
    cov_c0[5] = 0.00133768;
    cov_c0[6] = 0.00271096;
    cov_c0[7] = 0.00133768;
    cov_c0[8] = 0.00337321;
    CHECK_VECTOR_APPROX(a.models[label_a].components[0].covariance, cov_c0);
    a.reset();
    std::vector<double> log_likelihood(100, 0.0);
    for (unsigned int i = 0; i < 100; i++) {
        observation[0] = float(i) / 100.;
        observation[1] = pow(float(i) / 100., 2.);
        observation[2] = pow(float(i) / 100., 3.);
        a.filter(observation);
        log_likelihood[i] = a.results.smoothed_log_likelihoods[0];
    }
}

TEST_CASE("Cancel Training", "[GMM]") {
    xmm::TrainingSet ts(xmm::MemoryMode::OwnMemory,
                        xmm::Multimodality::Unimodal);
    ts.dimension.set(3);
    std::vector<float> observation(3);
    ts.addPhrase(0);
    ts.addPhrase(1);
    for (unsigned int i = 0; i < 100; i++) {
        observation[0] = float(i) / 100.;
        observation[1] = pow(float(i) / 100., 2.);
        observation[2] = pow(float(i) / 100., 3.);
        ts.getPhrase(0)->record(observation);
        ts.getPhrase(1)->record(observation);
    }
    std::string label_a(static_cast<std::string>("a"));
    std::string label_b(static_cast<std::string>("b"));
    ts.getPhrase(0)->label.set(label_a);
    ts.getPhrase(1)->label.set(label_b);
    xmm::GMM a;
    MyTrainingListener *listener = new MyTrainingListener();
    a.training_events.addListener(listener, &MyTrainingListener::onEvent);
    a.configuration.gaussians.set(3);
    a.configuration.multithreading = xmm::MultithreadingMode::Background;
    a.train(&ts);
    a.cancelTraining();
    CHECK(a.size() == 0);
    a.configuration.multithreading = xmm::MultithreadingMode::Parallel;
    a.train(&ts);
    a.cancelTraining();
    CHECK(a.size() == 2);
    a.configuration.multithreading = xmm::MultithreadingMode::Sequential;
    a.train(&ts);
    a.cancelTraining();
    CHECK(a.size() == 2);
}
